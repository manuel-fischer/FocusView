#!/usr/bin/env python
#
# Requires pdfcrop
#
# pip install pypdfium2
# pip install customtkinter
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Protocol, Set, Tuple
import customtkinter
from customtkinter import * # CTk*
import tkinter
from PIL import Image, ImageTk, ImageOps, ImageChops, ImageFilter

from functools import partial

import pypdfium2 as pdfium
import sys

import functools
import bisect
import json

import tempfile
import subprocess
import math


from weakref import WeakValueDictionary



def run_exec(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #if result.stdout or result.stderr or result.returncode:
    #    msg = f"stdout:\n{result.stdout.decode()}\n\nstderr:\n{result.stderr.decode()}\n\nreturncode:\n{result.returncode}"
    #    print(msg)
    #    tkinter.messagebox.showerror(message=msg)




def extract_filename(filename : str):
    fn, ext = os.path.splitext(filename)
    return f"{filename}-extract.pdf"


AUTO_RELOAD = True

Rect = Tuple[float, float, float, float]


clip_file = "{}-clip.json".format


class ClipContainer:
    subclips : 'List[Clip]'


    @classmethod
    def fromdict(cls, obj : dict):
        obj = obj.copy()
        if "subclips" in obj:
            obj["subclips"] = [Clip.fromdict(c) for c in obj["subclips"]]
        else:
            obj["subclips"] = []
        return cls(**obj)

class ClipContainerList(Protocol):
    root : 'ClipDocument'
    def __getitem__(self, page_number): ...
    def __iter__(self): ...


class FlattenedClipContainer(ClipContainer):
    subclips : 'List[Clip]'

    def __init__(self, subclips : 'List[Clip]'):
        self.subclips = subclips

# flatten by 1 level, do not keep clips in current
# return new ClipContainer and list of offsets for each page
def make_flattened_clip_container(clip_container_list: ClipContainerList) -> Tuple[ClipContainer, List[int]]:
    offset = 0
    clips = []
    offsets = []
    for c in clip_container_list:
        offsets.append(offset)
        if c is None: continue
        clips.extend(c.subclips)
        offset += len(c.subclips)

    # TODO: do not use base class
    return FlattenedClipContainer(subclips=clips), offsets



@dataclass
class ClipStyle:
    mark_color : Tuple[float, float, float]

    @classmethod
    def fromdict(cls, obj: dict):
        obj = dict(obj) # copy
        obj["mark_color"] = tuple(obj["mark_color"])
        return cls(**obj)

    def asdict(self):
        return asdict(self)

    def apply(self, img):
        R,G,B = self.mark_color
        R/=255;G/=255;B/=255
        filter = ImageFilter.Color3DLUT.generate(2, lambda r,g,b: (r*R,g*G,b*B), target_mode="RGB")
        return img.filter(filter)


@dataclass
class Clip(ClipContainer):
    rect : 'Rectangle' # Rect
    subclips : 'List[Clip]'
    style : "ClipStyle|None"

    @classmethod
    def fromdict(cls, obj: dict):
        obj = obj.copy()
        obj["rect"] = Rectangle.fromdict(obj["rect"])
        if "style" not in obj: obj["style"] = None
        elif obj["style"] is not None: obj["style"] = ClipStyle.fromdict(obj["style"])
        return super().fromdict(obj)

    def asdict(self):
        d = dict()
        PRECISION = 5
        P = 10**PRECISION
        d["rect"] = [math.floor(c*P+0.5)/P for c in self.rect]
        if self.subclips: d["subclips"] = [c.asdict() for c in self.subclips]

        if self.style != None:
            d["style"] = self.style.asdict()
        return d


@dataclass
class ClipPage(ClipContainer):
    page : int
    subclips : 'List[Clip]'

    def asdict(self):
        d = dict()
        d["page"] = self.page
        if self.subclips: d["subclips"] = [c.asdict() for c in self.subclips]
        return d


@dataclass
class ClipDocument(ClipContainerList):
    @property
    def root(self): return self

    filename : str
    pages : List[ClipPage]

    page_lookup : Dict[int, ClipPage] = field(metadata={"include_in_dict":False}, default_factory=lambda:None)

    def __build_lookup(self):
        self.page_lookup = {
            p.page: p for p in self.pages
        }

    def __getitem__(self, page_number):
        if self.page_lookup is None: self.__build_lookup()

        try: return self.page_lookup[page_number]
        except KeyError: pass

        page = ClipPage(page_number, [])
        self.page_lookup[page_number] = page
        #bisect.insort(self.pages, page, key=lambda p: p.page_number)
        self.pages.append(page)
        return page

    def asdict(self):
        #d = asdict(self)
        #d["pages"] = [p for p in d["pages"] if p["subclips"]] # remove empty pages
        #del d["filename"]
        #del d["page_lookup"]
        d = dict()
        d["pages"] = [p.asdict() for p in self.pages if p.subclips]
        return d

    @classmethod
    def fromdict(cls, obj : dict):
        obj = obj.copy()
        if "page_lookup" in obj: del obj["page_lookup"]
        obj["pages"] = [ClipPage.fromdict(p) for p in obj["pages"]]
        return cls(**obj)

    def __iter__(self): # fill empty ones with 0
        page = 0
        for p in sorted(self.pages, key=lambda p:p.page):
            for _ in range(p.page - page):
                yield None
            yield p
            page=p.page+1


@dataclass
class SingleClipContainerList(ClipContainerList):
    root : ClipDocument
    page : ClipContainer

    def __getitem__(self, page_number):
        assert page_number == 0
        return self.page

    def __iter__(self): yield self.page

@dataclass
class MultiClipContainerList(ClipContainerList):
    root : ClipDocument
    clip_container : ClipContainer

    def __getitem__(self, page_number):
        return self.clip_container.subclips[page_number]

    def __iter__(self): return iter(self.clip_container.subclips)

def adjust_rect(rect : Rect):
    x0, y0, x1, y1 = rect
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return x0, y0, x1, y1

MAX_WIDTH = 20000
MAX_HEIGHT = 10000


def notify_all(set_of_handlers: Set[Callable], *a, **kw):
    for h in set_of_handlers: h(*a, **kw)

RenderTime : type = float

class IDocument():
    name : str
    page_count : int
    on_reload : Set[Callable[[],None]]

    def __init__(self):
        self.on_reload = set()

    def needs_reload(self) -> bool: ...
    def needs_redraw(self, t: RenderTime) -> bool: ... # all results from render_page are not up to date
    def force_reload(self): ...
    def render_page(self, page_number: int, width : float, height : float, crop_rect : Rect = (0,0,1,1),**kw) -> Tuple[RenderTime, int, Image.Image]: ...

    #def register_reload_handler(self, handler: Callable[[],None]): self.on_reload.add(handler)
    def notify_reload_handler(self): notify_all(self.on_reload)

    def pdf_page(self, page_number: int, crop_rect: Rect) -> Tuple[pdfium.PdfDocument, int, Rect]: ...


class Cache:
    __dict : WeakValueDictionary
    __key_func : Callable[[any],any]

    __lru_keep : List
    __keep_index : int


    def __init__(self, max_lru : int = 128, key : Callable[[any],any] = lambda k:k):
        self.__dict = WeakValueDictionary()
        self.__key_func = key
        self.__lru_keep = [None]*max_lru
        self.__keep_index = 0

    def __setitem__(self, key, value):
        self.__dict[self.__key_func(key)] = value
        self.__lru_keep[self.__keep_index] = (key, value)
        self.__keep_index += 1
        self.__keep_index %= len(self.__lru_keep)

    def __getitem__(self, key):
        return self.__dict[self.__key_func(key)]

    def get(self, key, generate):
        try:
            return self[key]
        except KeyError:
            value = generate()
            self[key] = value
            return value


def rect_to_parent(parent_rect: Rect, crop_rect: Rect):
    x0, y0, x1, y1 = parent_rect
    xD, yD = x1-x0, y1-y0
    return (
        x0+xD*crop_rect[0], y0+yD*crop_rect[1],
        x0+xD*crop_rect[2], y0+yD*crop_rect[3],
    )


def render_page_clip(pdf_page, width: float, height: float, crop_rect: Rect, flip_lcd: bool = False, scale: "float|int" = 1) -> Image.Image:
    FLIP_LCD_ROT = 180
    rotation = FLIP_LCD_ROT if flip_lcd else 0

    pw, ph = pdf_page.get_size()

    crop = ( # bottom and top flipped
        pw*crop_rect[0], ph*(1-crop_rect[3]),
        pw*(1-crop_rect[2]), ph*crop_rect[1],
    )
    if flip_lcd:
        crop = (crop[2], crop[3], crop[0], crop[1])

    cw = (crop_rect[2]-crop_rect[0])*pw
    ch = (crop_rect[3]-crop_rect[1])*ph

    if width is None or height is None:
        pass #scale = kw.get("scale", 1)
    else:
        scale = min(width/cw if cw else 9999, height/ch if ch else 9999, 9999)
        scale = max(0.2, scale)
    print(pdf_page.get_rotation())

    print(crop)

    img = pdf_page.render_topil(
        scale=scale,
        rotation=rotation,
        crop=crop,
        greyscale=False,
        optimise_mode=pdfium.OptimiseMode.LCD_DISPLAY if USE_LCD else pdfium.OptimiseMode.NONE,
    )

    if flip_lcd:
        img = ImageOps.flip(ImageOps.mirror(img))

    return img


class PDFDocument(IDocument):
    filename : str
    modify_time : "float | None"
    pdf : "pdfium.PdfDocument | None"
    page_count : int

    cache : Cache #[Tuple[int, float, float, Rect], Image.Image]

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.modify_time = None
        self.pdf = None
        self.page_count = 0
        self.force_reload()


    def needs_reload(self) -> bool:
        return os.path.getmtime(self.filename) > self.modify_time

    def needs_redraw(self, t: RenderTime) -> bool:
        return self.needs_reload() or self.modify_time > t

    def force_reload(self):
        self.modify_time = os.path.getmtime(self.filename)
        if self.pdf: self.pdf.close()
        self.pdf = pdfium.PdfDocument(self.filename)
        self.page_count = len(self.pdf)
        assert self.pdf
        self.cache = Cache()
        self.notify_reload_handler()

    def render_page(self, page_number: int, width: float, height: float, crop_rect: Rect = (0,0,1,1),**kw) -> Tuple[RenderTime, int, Image.Image]:
        if AUTO_RELOAD and self.needs_reload(): self.force_reload()

        if page_number >= self.page_count: page_number = self.page_count-1

        flip_lcd = kw.get("flip_lcd", False)
        scale = kw.get("scale", 1)

        param_tuple = page_number,width,height,crop_rect,flip_lcd,scale

        try: return (self.modify_time, page_number, self.cache[param_tuple])
        except KeyError: pass


        page = self.pdf.get_page(page_number)

        img = render_page_clip(page, width, height, crop_rect, flip_lcd, scale)

        self.cache[param_tuple] = img

        return (self.modify_time, page_number, img)


    def pdf_page(self, page_number: int, crop_rect: Rect) -> Tuple[pdfium.PdfDocument, int, Rect]:
        return (self.pdf, page_number, crop_rect)


# just 1 page
class SubpageDocument(IDocument):
    document : IDocument
    page_number : int
    rect : Rect
    page_count : int = 1

    def __init__(self, document: IDocument, page_number: int, rect: Rect) -> None:
        super().__init__()
        self.document = document
        self.page_number = page_number
        self.rect = adjust_rect(rect)


    def needs_reload(self) -> bool:
        return self.document.needs_reload()

    def needs_redraw(self, t: RenderTime) -> bool:
        return self.document.needs_redraw(t)

    def force_reload(self):
        return self.document.force_reload()

    def render_page(self, page_number: int, width: float, height: float, crop_rect: Rect = (0,0,1,1),**kw) -> Tuple[RenderTime, int, Image.Image]:
        assert page_number == 0

        crop_rect2 = rect_to_parent(self.rect, crop_rect)

        # TODO: handle changed page number
        render_time, inner_page_number, image = self.document.render_page(self.page_number, width, height, crop_rect2,**kw)
        return render_time, 0, image

    def pdf_page(self, page_number: int, crop_rect: Rect) -> Tuple[pdfium.PdfDocument, int, Rect]:
        assert page_number == 0
        crop_rect2 = rect_to_parent(self.rect, crop_rect)
        return self.document.pdf_page(self.page_number, crop_rect2)


class MultiSubpageDocument(IDocument):
    document : IDocument
    page_number : int
    clip_container : ClipContainer

    def __init__(self, document: IDocument, page_number: int, clip_container: ClipContainer):
        self.document = document
        self.page_number = page_number
        self.clip_container = clip_container

    @property
    def page_count(self): return len(self.clip_container.subclips)


    def needs_reload(self) -> bool:
        return self.document.needs_reload()

    def needs_redraw(self, t: RenderTime) -> bool:
        return self.document.needs_redraw(t)

    def force_reload(self):
        return self.document.force_reload()

    def render_page(self, page_number: int, width: float, height: float, crop_rect: Rect = (0,0,1,1),**kw) -> Tuple[RenderTime, int, Image.Image]:
        page_number = min(max(page_number, 0), self.page_count)

        crop_rect2 = rect_to_parent(self.clip_container.subclips[page_number].rect, crop_rect)

        # TODO: handle changed page number
        render_time, inner_page_number, image = self.document.render_page(self.page_number, width, height, crop_rect2,**kw)
        return render_time, page_number, image

    def pdf_page(self, page_number: int, crop_rect: Rect) -> Tuple[pdfium.PdfDocument, int, Rect]:
        crop_rect2 = rect_to_parent(self.clip_container.subclips[page_number].rect, crop_rect)
        return self.document.pdf_page(self.page_number, crop_rect2)



class FlattenedMultiSubpageDocument(IDocument):
    document : IDocument
    page_number_offsets : List[int]
    clip_container : ClipContainer

    def __init__(self, document: IDocument, page_number_offsets: int, clip_container: ClipContainer):
        self.document = document
        self.page_number_offsets = page_number_offsets
        self.clip_container = clip_container

    @property
    def page_count(self): return len(self.clip_container.subclips)

    def needs_reload(self) -> bool:
        return self.document.needs_reload()

    def needs_redraw(self, t: RenderTime) -> bool:
        return self.document.needs_redraw(t)

    def force_reload(self):
        return self.document.force_reload()

    def __translate_page_number(self, page_number: int):
        page_number = min(max(page_number, 0), self.page_count)
        return max(bisect.bisect_right(self.page_number_offsets, page_number)-1, 0)

    def render_page(self, page_number: int, width: float, height: float, crop_rect: Rect = (0,0,1,1),**kw) -> Tuple[RenderTime, int, Image.Image]:

        x0, y0, x1, y1 = self.clip_container.subclips[page_number].rect
        xD, yD = x1-x0, y1-y0

        crop_rect2 = rect_to_parent(self.clip_container.subclips[page_number].rect, crop_rect)

        # TODO: handle changed page number
        parent_page_number = self.__translate_page_number(page_number)
        render_time, inner_page_number, image = self.document.render_page(parent_page_number, width, height, crop_rect2,**kw)
        return render_time, page_number, image

    def pdf_page(self, page_number: int, crop_rect: Rect) -> Tuple[pdfium.PdfDocument, int, Rect]:
        crop_rect2 = rect_to_parent(self.clip_container.subclips[page_number].rect, crop_rect)
        parent_page_number = self.__translate_page_number(page_number)
        return self.document.pdf_page(parent_page_number, crop_rect2)




customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")


def CTkButtonWith(master, **k):
    def inner(command):
        return CTkButton(master=master, **k, command=command)

    return inner

def bind(object, func_name):
    def inner(event_handler):
        object.bind(func_name, event_handler)
        return event_handler
    return inner

def bind_all(object, func_name):
    def inner(event_handler):
        object.bind_all(func_name, event_handler)
        return event_handler
    return inner



clamp = lambda x, a, b: min(max(x, a), b)

def rgb_space(func):
    #return func

    GAMMA = 2.2
    GAMMA = 1.5
    GAMMA = 1.3
    def map(r,g,b):
        r**=GAMMA
        g**=GAMMA
        b**=GAMMA
        r,g,b = func(r,g,b)
        r, g, b = max(r,0), max(g,0), max(b,0)
        r**=1/GAMMA
        g**=1/GAMMA
        b**=1/GAMMA
        return r,g,b

    return map
#N = 2
N = 32
invert_dark_mode_filter = ImageFilter.Color3DLUT.generate(
    size=N,
    callback=rgb_space(lambda r,g,b: [
        clamp(1-(g+b-r), 0, 1),
        clamp(1-(b+r-g), 0, 1),
        clamp(1-(r+g-b), 0, 1),
    ]),
    target_mode="RGB"
)
invert_dark_mode_vibrant_filter = ImageFilter.Color3DLUT.generate(
    size=N,
    callback=rgb_space(lambda r,g,b: [
        1-(g+b-r),
        1-(b+r-g),
        1-(r+g-b),
    ]),
    target_mode="RGB"
)
invert_reduced_blue_mode_filter = ImageFilter.Color3DLUT.generate(
    size=N,
    callback=rgb_space(lambda r,g,b: [
        (1-(g+b-r))*1.1,
        (1-(b+r-g))*0.9,
        (1-(r+g-b))*0.6,
    ]),
    target_mode="RGB"
)
invert_dark_mode_low_filter = ImageFilter.Color3DLUT.generate(
    size=N,
    callback=rgb_space(lambda r,g,b: [
        (1/16) + (11/16)*(1+r-g-b),
        (1/16) + (11/16)*(1+g-b-r),
        (1/16) + (11/16)*(1+b-r-g),
    ]),
    target_mode="RGB"
)
reduced_blue_mode_filter = ImageFilter.Color3DLUT.generate(
    size=N,
    callback=rgb_space(lambda r,g,b: [
        r*1.1,
        g*0.9,
        b*0.6
    ]),
    target_mode="RGB"
)


class HackersMode:
    blur_gamma = ImageFilter.Color3DLUT.generate(
        size=10,
        callback=lambda r,g,b: [
            r**0.4,#0.2,
            g**0.4,#0.2,
            b**0.4,#0.2,
        ],
        target_mode="RGB"
    )
    light_up_filter = ImageFilter.Color3DLUT.generate(
        size=10,
        callback=lambda r,g,b: [
            r*0.75+(g+b)*0.25,#(r+g+b)/2,
            g*0.75+(b+r)*0.25,#(r+g+b)/2,
            b*0.75+(r+g)*0.25,#(r+g+b)/2,
        ],
        target_mode="RGB"
    )
    blur_filter = ImageFilter.GaussianBlur(radius=10)
    blur_filter2 = ImageFilter.GaussianBlur(radius=3)

    def __init__(self, rf, gf, bf, light_up=False, glow_factor=0.25):
        self.color_filter = ImageFilter.Color3DLUT.generate(
            size=N,
            callback=lambda r,g,b: [
                (1-(g+b-r))*rf,
                (1-(b+r-g))*gf,
                (1-(r+g-b))*bf,
            ],
            target_mode="RGB"
        )
        self.blur_scale = ImageFilter.Color3DLUT.generate(
            size=10,
            callback=lambda r,g,b: [
                0.75*glow_factor*(2*r-0.5*(g+b)),
                0.75*glow_factor*(2*g-0.5*(b+r)),
                0.75*glow_factor*(2*b-0.5*(r+g)),
            ],
            target_mode="RGB"
        )
        self.blur_scale2 = ImageFilter.Color3DLUT.generate(
            size=10,
            callback=lambda r,g,b: [
                1.5*glow_factor*(2*r-0.5*(g+b)),
                1.5*glow_factor*(2*g-0.5*(b+r)),
                1.5*glow_factor*(2*b-0.5*(r+g)),
            ],
            target_mode="RGB"
        )
        self.light_up = light_up

    def do_filter(self, i : Image.Image):
        i = i.filter(self.color_filter)

        bi = i

        bi = bi.filter(self.blur_filter)
        bi = bi.filter(self.blur_gamma)
        bi = bi.filter(self.blur_scale)

        bi2 = i
        bi2 = bi2.filter(self.blur_filter2)
        bi2 = bi2.filter(self.blur_scale2)
        bi = ImageChops.add(bi, bi2)
        if self.light_up: i = i.filter(self.light_up_filter)
        i = ImageChops.add(adjust_lcd(i), bi)

        return i


SHADOW_FACTOR = 0.4
class ShadowMode:
    blur_scale = ImageFilter.Color3DLUT.generate(
        size=5,
        callback=lambda r,g,b: [
            1 + SHADOW_FACTOR*(r-1),
            1 + SHADOW_FACTOR*(g-1),
            1 + SHADOW_FACTOR*(b-1),
        ],
        target_mode="RGB"
    )
    gray_scale = ImageFilter.Color3DLUT.generate(
        size=5,
        callback=lambda r,g,b: [
            (r+g+b)*0.334,
            (r+g+b)*0.334,
            (r+g+b)*0.555,
        ],
        target_mode="RGB"
    )
    blur_filter = ImageFilter.GaussianBlur(radius=7)

    @classmethod
    def do_filter(cls, i : Image.Image):
        bi = shift_img_xy(i, 5, 5, (255, 255, 255))
        bi = bi.filter(cls.gray_scale)
        bi = bi.filter(cls.blur_filter)
        bi = bi.filter(cls.blur_scale)
        #i = ImageChops.multiply(i, bi)
        i = ImageChops.darker(i, bi)

        return i


def shift_img(img, dx):
    n = Image.new(mode=img.mode, size=img.size)
    n.paste(img, (dx, 0))
    return n

def shift_img_xy(img, dx, dy, color):
    n = Image.new(mode=img.mode, size=img.size, color=color)
    n.paste(img, (dx, dy))
    return n

USE_LCD = True

def adjust_lcd(img):
    return img

    if not USE_LCD: return img

    """shifts red and blue around to fix lcd rendering"""

    r, g, b = img.convert("RGB").split()

    r=shift_img(r, +1)
    b=shift_img(b, -1)

    return Image.merge("RGB",(r,g,b))

@dataclass
class VisualEffect:
    effect_func : "Callable[[Image.Image], Image.Image] | None"
    render_args : dict

DARK_ARGS = {"flip_lcd": True}

visual_effects_list = [
    ("#000000", VisualEffect(lambda i: adjust_lcd(i.filter(invert_reduced_blue_mode_filter)), DARK_ARGS)),
    ("#000000", VisualEffect(lambda i: adjust_lcd(i.filter(invert_dark_mode_vibrant_filter)), DARK_ARGS)),
    ("#1e1e1e", VisualEffect(lambda i: adjust_lcd(i.filter(invert_dark_mode_low_filter)), DARK_ARGS)),
    #("#000000", VisualEffect(lambda i: adjust_lcd(i.filter(invert_dark_mode_filter)), DARK_ARGS)),
    ("#ffffff", VisualEffect(None, {})),
    ("#ffebac", VisualEffect(lambda i: i.filter(reduced_blue_mode_filter), {})),
    ("#ffffff", VisualEffect(ShadowMode.do_filter, {})),
    ("#000000", VisualEffect(HackersMode(1,1,1,True, 0.5).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(0.7,1.1,0.3).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(1.1,0.3,1.1).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(1.1,0.8,0.2).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(1.1,0.5,0.1).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(1.0,0.05,0.2).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(0.1,0.1,1.0).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(0.2,0.9,1.0).do_filter, DARK_ARGS)),
    ("#000000", VisualEffect(HackersMode(0.1,1.0,0.3).do_filter, DARK_ARGS)),
]
background_colors, visual_effects = [v[0] for v in visual_effects_list], [v[1] for v in visual_effects_list]
DEFAULT_VISUAL_EFFECT = 2 # 5


glb_image_effect_cache = Cache(key=lambda fx_img: (id(fx_img[0]), id(fx_img[1])))
glb_photo_image_cache = Cache(key=lambda img: id(img))

def apply_effect_cached(fx, image):
    if fx is not None:
        image = glb_image_effect_cache.get((fx, image), generate=lambda: fx(image))
    return image

class ImageViewer(CTkFrame):
    # canvas, image_rect, dark_mode

    def __init__(self, master, image_func, visual_effect_func = None, render_args={}): # TODO: render_args: dark, rotated rendering by 180 degree for better lcd rendering
        super().__init__(master=master)
        self.image_func = image_func

        BG_COLOR = "#000000" # "#181818"

        self.canvas = CTkCanvas(master=self)
        self.canvas.configure(background=BG_COLOR, highlightthickness=0)
        self.canvas.pack(expand=True, fill=BOTH)
        self.canvas.bind("<Configure>", self.canvas_configure_hook)

        self.image_rect = None
        self.visual_effect_func = visual_effect_func
        self.render_args = render_args
        self.fraction = 1
        self.pos_offset = (0, 0)

        self.__ref_draw = None
        self.original_image = None
        self.__ph_img = None

    def canvas_configure_hook(self, event):
        self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.update()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

        #if self.__image is not None: # fast preview
        #    self.canvas.delete(self.__ref_draw)
        #    iw, ih = self.__image.size
        #    scale = min(w/iw, h/ih)
        #    image = self.__image.resize((int(iw*scale), int(ih*scale)))
        #    self.__ph_img = ImageTk.PhotoImage(image)
        #    self.__ref_draw = self.canvas.create_image(w//2, h//2, image=self.__ph_img)
        #    self.canvas.update()

        image = self.image_func(w*self.fraction, h*self.fraction,**self.render_args)
        self.original_image = image
        image = apply_effect_cached(self.visual_effect_func, image)

        x, y = (w-image.width)//2, (h-image.height)//2
        x += self.pos_offset[0]
        y += self.pos_offset[1]
        self.image_rect = (
            x, y,
            x+image.width, y+image.height,
        )

        self.canvas.delete(self.__ref_draw)
        self.__ph_img = glb_photo_image_cache.get(image, generate=lambda:ImageTk.PhotoImage(image))
        #self.__ph_img = ImageTk.PhotoImage(image)
        #self.__ref_draw = self.canvas.create_image(w//2, h//2, image=self.__ph_img)
        self.__ref_draw = self.canvas.create_image(x, y, image=self.__ph_img, anchor="nw")

    def image_to_canvas(self, coords):
        if self.image_rect is None: return coords
        x0, y0, x1, y1 = self.image_rect

        translate = (
            lambda x: x0 + (x1-x0)*x,
            lambda y: y0 + (y1-y0)*y,
        )

        return [translate[i&1](c) for i, c in enumerate(coords)]

    def canvas_to_image(self, coords):
        if self.image_rect is None: return coords
        x0, y0, x1, y1 = self.image_rect

        translate = (
            lambda x: (x-x0)/(x1-x0),
            lambda y: (y-y0)/(y1-y0),
        )

        return [translate[i&1](c) for i, c in enumerate(coords)]



@dataclass
class Rectangle:
    x0 : float; y0 : float
    x1 : float; y1 : float

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = adjust_rect((x0,y0,x1,y1))

    def __iter__(self):
        return iter([self.x0, self.y0, self.x1, self.y1])

    def hit_test(self, x, y) -> bool:
        x0, x1 = min(self.x0, self.x1), max(self.x0, self.x1)
        y0, y1 = min(self.y0, self.y1), max(self.y0, self.y1)
        return x0 <= x < x1 and \
               y0 <= y < y1


    @property
    def width(self): return abs(self.x0-self.x1)

    @property
    def height(self): return abs(self.y0-self.y1)

    @classmethod
    def fromdict(cls, obj : "dict|list"):
        if type(obj) is list: return cls(*obj)
        return cls(**obj)

class PageSelector(ImageViewer):
    __rects : List[Rectangle]
    __rect  : Rectangle


    def __init__(self, master, image_func, render_args:dict={},
                 on_rect_add = (lambda index, rect, event: None),
                 on_rect_remove = (lambda index, rect: None),
                 on_rect_click = (lambda event, index, rect: None), **k):
        super().__init__(master=master, image_func=image_func, render_args=render_args, **k)

        self.canvas.bind("<Button-1>", self.b1_down)
        self.canvas.bind("<B1-Motion>", self.b1_move)
        self.canvas.bind("<Button-3>", self.b3_down)
        self.canvas.bind("<ButtonRelease-1>", self.b1_up)
        self.canvas.bind("<Motion>", self.m_move)

        self.canvas.configure(cursor="crosshair")

        self.on_rect_add = on_rect_add
        self.on_rect_remove = on_rect_remove
        self.on_rect_click = on_rect_click

        self.get_custom_effect_hook = None

        self.__sel_rect = None
        self.__rects = []
        self.__rects_ids = [] # ids + photo-images

        self.__rect = None
        self.__rect_ids = [] #{} # map id(rect) to array of ids

        self.__drawing = False

    def __custom_rect_effect(self, index):
        if self.get_custom_effect_hook is None: return None
        return self.get_custom_effect_hook(index)

    # return new rect_ids
    def __redraw_rect(self, rect, rect_ids, selected = False, effect = None):
        if rect_ids:
            rect_ids = [i for i in rect_ids if type(i) != ImageTk.PhotoImage]
            self.canvas.delete(*rect_ids)

        if rect is None: return

        style = dict(width = 3) if selected else {}



        # effects
        #effect = lambda img: ImageChops.invert(img)

        c_rect = self.image_to_canvas(rect)
        ids = []
        if effect and self.original_image:
            x0,y0,x1,y1 = c_rect
            x,y=x0,y0
            x0-=self.image_rect[0]
            y0-=self.image_rect[1]
            x1-=self.image_rect[0]
            y1-=self.image_rect[1]
            img = self.original_image.crop((x0,y0,x1,y1))
            img = effect(img)
            if self.visual_effect_func: img=self.visual_effect_func(img)
            phimg = ImageTk.PhotoImage(img)
            ids.append(phimg)
            ids.append(self.canvas.create_image(x, y, image=phimg, anchor="nw"))

        if not effect or selected:
            ids.append(self.canvas.create_rectangle(*c_rect, **style, outline="#808080"))#, dash=(4,4)))
            #ids.append(self.canvas.create_rectangle(*c_rect, **style, outline="white"))
            #ids.append(self.canvas.create_rectangle(*c_rect, **style, outline="black", outlinestipple="gray50"))

        return ids

    def __redraw_sel_rect(self):
        self.__rect_ids = self.__redraw_rect(self.__rect, self.__rect_ids, selected=False, effect=None)

    def __redraw_rects_rect(self, index):
        self.__rects_ids[index] = self.__redraw_rect(
            self.__rects[index], self.__rects_ids[index],
            selected=self.__sel_rect==index,effect=self.__custom_rect_effect(index))

    def __redraw_rects(self):
        new_rects_ids = []
        assert len(self.__rects) == len(self.__rects_ids)
        for i, (rect, rect_ids) in enumerate(zip(self.__rects, self.__rects_ids)):
            new_rects_ids.append(
                self.__redraw_rect(rect, rect_ids, selected=self.__sel_rect==i, effect=self.__custom_rect_effect(i))
            )
        self.__rects_ids = new_rects_ids

    def redraw_canvas(self):
        super().redraw_canvas()

        self.__redraw_sel_rect()
        self.__redraw_rects()


    def m_move(self, event):
        sel = self.spy_rect(event.x, event.y)

        cursor = "hand2" if sel is not None else "crosshair"
        self.canvas.configure(cursor=cursor)


    def __b1_down__start_draw(self, event):
        self.__rect = tuple([*self.canvas_to_image((event.x, event.y))]*2)
        self.__redraw_sel_rect()
        self.__drawing = True

    def b1_down(self, event):
        sel = self.spy_rect(event.x, event.y)
        self.__drawing = False
        if sel is None: self.__b1_down__start_draw(event)
        elif sel == self.__sel_rect: self.on_rect_click(event, sel, self.__rects[sel])
        else: self.select_rect(sel)

    def b1_move(self, event):
        if not self.__drawing: return

        self.__rect = self.__rect[:2] + tuple([*self.canvas_to_image((event.x, event.y))])
        self.__redraw_sel_rect()

    def b1_up(self, event):
        if not self.__drawing: return

        # if the rectangle is just a few pixels large, discard it
        ir = self.__rect
        ir = [min(max(c, 0), 1) for c in ir]
        r = Rectangle(*self.image_to_canvas(ir))
        if r.width > 3 or r.height > 3:
            self.add_rect_with_event(ir, select=True, event=event)
        else:
            self.select_rect(None)

        self.__rect = None
        self.__redraw_sel_rect()


    def b3_down(self, event):
        self.__rect = None
        self.__redraw_sel_rect()

        self.select_rect(None)


    @property
    def rect(self): return self.__rect

    @property
    def rects(self): return self.__rects

    @rects.setter
    def rects(self, new_rects):
        for rect_id in self.__rects_ids:
            self.__redraw_rect(None, rect_id)

        self.__sel_rect = None
        self.__rects = [
            Rectangle(*r)
            for r in new_rects
        ]
        new_rects_ids = []
        for i,rect in enumerate(self.__rects):
            new_rects_ids.append(
                self.__redraw_rect(rect, None, selected=False, effect=self.__custom_rect_effect(i))
            )
        self.__rects_ids = new_rects_ids



    @property
    def selection(self): return self.__sel_rect

    def select_rect(self, index : "int | None"):
        if self.__sel_rect == index: return

        old_sel = self.__sel_rect
        self.__sel_rect = index

        if old_sel is not None: self.__redraw_rects_rect(old_sel)
        if index is not None:   self.__redraw_rects_rect(index)


    def set_rect(self, index, rect):
        self.__rects[index] = Rectangle(*rect)
        self.__redraw_rects_rect(index)

    def add_rect_with_event(self, rect, select=False, event=None):
        self.add_rect(rect,select)
        self.on_rect_add(len(self.__rects)-1, rect, event)

    def add_rect(self, rect, select=False):
        if rect is None: return

        if select:
            self.select_rect(None)
            self.__sel_rect = len(self.__rects)

        rect = Rectangle(*rect)
        self.__rects.append(rect)
        self.__rects_ids.append(
            self.__redraw_rect(rect, None, selected=select)
        )

    def refresh_rect(self, index : int):
        self.__rects_ids[index] = self.__redraw_rect(self.__rects[index], self.__rects_ids[index], selected=self.selection==index, effect=self.__custom_rect_effect(index))

    def remove_rect(self, index : int):
        if index is None: return

        self.__redraw_rect(None, self.__rects_ids[index])
        self.on_rect_remove(index, self.__rects[index])
        del self.__rects[index]
        del self.__rects_ids[index]
        if self.__sel_rect is not None:
            if self.__sel_rect == index: self.__sel_rect = None
            elif self.__sel_rect > index: self.__sel_rect -= 1

    # gets rectangle at canvas coordinates
    def spy_rect(self, cx, cy) -> "int | None":
        x, y = self.canvas_to_image([cx, cy])
        for i, r in enumerate(reversed(self.__rects)):
            if r.hit_test(x, y):
                return len(self.__rects) - i - 1
        return None



class PDFPage(Protocol):
    #page_size_func : Callable[[], Tuple[float, float]]
    def needs_redraw(self) -> bool: ...
    def image_func(self, width: float, height: float, crop_rect: Rect,**kw)-> Image.Image: ...




class DynamicDocumentPage(PDFPage):
    document: IDocument
    page_number: int
    render_time: RenderTime

    on_page_changed : Set[Callable[[int], None]]

    def __init__(self, document: IDocument, page_number: int) -> None:
        super().__init__()
        self.document = document
        self.page_number = page_number
        self.render_time = 0
        self.on_page_changed = set()

    def needs_redraw(self) -> bool:
        return self.document.needs_redraw(self.render_time)

    def image_func(self, width: float, height: float, crop_rect: Rect=(0,0,1,1),**kw) -> Image.Image:
        render_time, page_number, image = self.document.render_page(
            page_number=self.page_number,
            width=width,
            height=height,
            crop_rect=crop_rect,**kw)
        if page_number != self.page_number:
            old_page_number = self.page_number
            self.page_number = page_number
            notify_all(self.on_page_changed, old_page_number)

        self.render_time = render_time

        return image

    def set_page(self, page_number : int) -> bool:
        """Set new page number, return true, if actually switched to a new page"""

        page_number = max(page_number, 0)
        page_number = min(page_number, self.document.page_count-1)

        if self.page_number == page_number: return False

        old_page_number = self.page_number
        self.page_number = page_number
        notify_all(self.on_page_changed, old_page_number)
        return True




icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_view_icon.png")

def prod(*iter):
    if not iter:
        yield ()
        return
    for a in iter[0]:
        for b in prod(*iter[1:]):
            yield (a,*b)

icons = None
def get_icons():
    global icons
    if icons: return icons

    icon_img = Image.open(icon_path).convert("RGBA")
    icons = [
        ImageTk.PhotoImage(icon_img.resize((1<<i,1<<i),Image.BOX))
        for i in range(4,8)
    ]

    #icons = [ImageTk.PhotoImage(icon_img)]
    #id = icon_img.load()
    #w = icon_img.width
    #for i in range(4):
    #    w //= 2
    #    print(w)
    #    i2 = Image.new("RGBA", (w,w), color=(0, 0, 255, 255))
    #    id2 = i2.load()
    #    for y in range(w):
    #        for x in range(w):
    #            c = (255, 255, 255, 0)
    #            for i,j in ((0,0),(0,1),(1,0),(1,1)):
    #                r0,g0,b0,a0 = c
    #                r1,g1,b1,a1 = id[x*2+i,y*2+j]
    #                c = (min(r0,r1), min(g0,g1), min(b0,b1), max(a0,a1))
    #
    #            id2[x,y] = c #tuple(map(min, zip(*colors)))
    #
    #    id=id2
    #    icons.append(ImageTk.PhotoImage(i2))

    icons.sort(key=lambda i:i.width())

    return icons


def translate_like(r, src, dst):
    rx0, ry0, rx1, ry1 = r
    sx0, sy0, sx1, sy1 = src
    dx0, dy0, dx1, dy1 = dst

    x_transform = lambda x: (x-sx0)*(dx1-dx0)/(sx1-sx0)+dx0
    y_transform = lambda y: (y-sy0)*(dy1-dy0)/(sy1-sy0)+dy0

    return x_transform(rx0), y_transform(ry0), x_transform(rx1), y_transform(ry1)



def pixel_align_rect(pdf_page_size, scale, rect):
    w, h = pdf_page_size
    w*=scale; h*=scale
    x0,y0,x1,y1 = rect
    return (
        int(math.floor(x0*w))/w, int(math.floor(y0*h))/h,
        int(math.ceil(x1*w))/w, int(math.ceil(y1*h))/h,
    )

def color_distance(a, b):
    return sum((ai-bi)**2 for ai,bi in zip(a,b))

# return coordinates in 0 to 1
def get_minimal_bounds(img):
    dyn_bg = img.load()[0,0]
    print("COLOR:", dyn_bg)

    accuracy = 32
    map_distance = ImageFilter.Color3DLUT.generate(
        size=accuracy,
        callback=lambda r,g,b: 3*[color_distance(dyn_bg, (r*255,g*255,b*255))/255/255],
        target_mode="RGB"
    )

    img = img.filter(map_distance)
    img.save("tmp_image.png")

    BG = (0,0,0)
    F = 1

    def is_background(c):
        return color_distance(c, BG) < 32**2

    def reduce_x(img):
        while img.width > 1:
            new_size = ((img.width+1)//2,img.height)
            imgA = Image.new(img.mode, new_size, color=BG)
            imgA.paste(img)
            imgB = Image.new(img.mode, new_size, color=BG)
            imgB.paste(img, (-img.width//2,0))
            img = ImageChops.lighter(imgA, imgB)
        return img
    def reduce_y(img):
        while img.height > 1:
            new_size = (img.width,(img.height+1)//2)
            imgA = Image.new(img.mode, new_size, color=BG)
            imgA.paste(img)
            imgB = Image.new(img.mode, new_size, color=BG)
            imgB.paste(img, (0,-img.height//2))
            img = ImageChops.lighter(imgA, imgB)
        return img

    x_strip = reduce_y(img).load()
    y_strip = reduce_x(img).load()

    for x_min in range(img.width):
        if not is_background(x_strip[x_min,0]):
            break

    for y_min in range(img.height):
        if not is_background(y_strip[0,y_min]):
            break

    for x_max in range(img.width,0,-1):
        if not is_background(x_strip[x_max-1,0]):
            break

    for y_max in range(img.height,0,-1):
        if not is_background(y_strip[0,y_max-1]):
            break


    if x_min > x_max or y_min > y_max: return (0,0,1,1)

    x_min = max(0,x_min-F)
    y_min = max(0,y_min-F)
    x_max = min(img.width,x_max+F)
    y_max = min(img.height,y_max+F)

    return x_min/img.width, y_min/img.height, x_max/img.width, y_max/img.height






SHIFT_STATE_MASK = 0x0001 # TODO check
CONTROL_STATE_MASK = 0x0004
ALT_STATE_MASK = 0x0008

global_root = None
def show_image(name, document: IDocument, clip_pages: ClipContainerList, visual_effect_index=DEFAULT_VISUAL_EFFECT, page_number: int=0):
    global global_root

    dpage = DynamicDocumentPage(document, page_number)

    topmost = False
    TOPFMT = "–   {}   –".format
    str_page_number = lambda: f"{dpage.page_number+1}/{document.page_count}"
    str_title = lambda topmost_mask=True: [str, TOPFMT][topmost and topmost_mask](f"{name}   –   {str_page_number()}")

    if global_root is None:
        global_root = root = CTk()
        #root.attributes("-zoomed", True)
    else:
        root = CTkToplevel(global_root)

    #root.iconbitmap(get_icon())
    root.iconphoto(True, *get_icons())
    root.geometry("1000x600")
    root.title(str_title())

    page_viewer = PageSelector(master=root, image_func=dpage.image_func,
        visual_effect_func=visual_effects[visual_effect_index].effect_func,render_args=visual_effects[visual_effect_index].render_args)
    page_viewer.canvas.configure(background=background_colors[visual_effect_index])
    page_viewer.pack(side = TOP, expand=True, fill=BOTH)

    #lbl_page_number = CTkLabel(master=root, text=str_page_number())
    #lbl_page_number.pack()

    #command_box = CTkTextbox(master=root)
    #command_box.pack(expand=Y, fill=X)


    def set_page(*, delta = None, index = None):
        if index is None:
            assert delta is not None
            index = dpage.page_number + delta
        else:
            assert delta is None

        if dpage.set_page(index):
            page_viewer.redraw_canvas()
            root.title(str_title())
            return True
        return False

    def set_effect_mode(index):
        nonlocal visual_effect_index
        visual_effect_index = index % len(visual_effects)
        page_viewer.visual_effect_func = visual_effects[visual_effect_index].effect_func
        page_viewer.render_args  = visual_effects[visual_effect_index].render_args
        page_viewer.redraw_canvas()
        page_viewer.canvas.configure(background=background_colors[visual_effect_index])

    def update_viewer_rects():
        page_viewer.rects = [
            p.rect
            for p in clip_pages[dpage.page_number].subclips
        ]
    update_viewer_rects()

    @dpage.on_page_changed.add
    def dpage_on_page_changed(old_page): update_viewer_rects()

    # TODO
    #current_style = ClipStyle(mark_color=(1,0.4,0.2))
    current_style = None

    @partial(setattr, page_viewer, "on_rect_add")
    def page_viewer_on_rect_add(index, rect, event):
        clip_pages[dpage.page_number].subclips.insert(index, Clip(rect, [], current_style))
        if not (event.state & CONTROL_STATE_MASK):
            shrink_to_fit_idx(index)
        else:
            # update style
            page_viewer.refresh_rect(index)

    @partial(setattr, page_viewer, "on_rect_remove")
    def page_viewer_on_rect_remove(index, rect):
        del clip_pages[dpage.page_number].subclips[index]

    @partial(setattr, page_viewer, "get_custom_effect_hook")
    def page_viewer_get_custom_effect(index):
        clip = clip_pages[dpage.page_number].subclips[index]
        if clip.style: return clip.style.apply
        return None


    def hex_to_float(hex):
        r = int(hex[1:3], 16)
        g = int(hex[3:5], 16)
        b = int(hex[5:7], 16)
        return r,g,b

    STYLE_COLORS = list(map(hex_to_float, [
        "#ffc0c0",
        "#ffe0c0",
        "#ffffc0",
        "#c0ffc0",
        "#c0ffff",
        "#c0c0ff",
        "#ffc0ff",
        "#c0c0c0",
        "#e0e0e0",
    ]))

    def set_style(i, event):
        if i == 0:
            new_style = None
        else:
            new_style = ClipStyle(STYLE_COLORS[i-1])


        sel = page_viewer.selection
        if sel is None:
            nonlocal current_style
            current_style = new_style
        else:
            clip_pages[dpage.page_number].subclips[sel].style = new_style
            page_viewer.refresh_rect(sel)

    for i in range(0, 10):
        bind(root, str(i))(partial(set_style, i))


    @bind(root, "<F12>")
    def toggle_topmost(event):
        nonlocal topmost
        topmost ^= True
        root.attributes("-topmost", topmost)
        root.title(str_title())
        root.update()

    @bind(root, "t")
    def toggle_dark_mode(event):
        set_effect_mode(visual_effect_index+1)

    @bind(root, "T")
    def toggle_dark_mode(event):
        set_effect_mode(visual_effect_index-1)


    #"""
    background_color_index = 2

    @bind(root, "b")
    def toggle_background_color(event):
        nonlocal background_color_index
        background_color_index += 1
        background_color_index %= 3

        if background_color_index == 2:
            c = background_colors[visual_effect_index]
        else:
            c = ["black", "white"][background_color_index]

        page_viewer.canvas.configure(background=c)
    #"""

    quit_time = None
    quit_count = 0
    @bind(root, "q")
    def quit(event):
        nonlocal quit_time, quit_count
        if global_root == root: # requires double q
            import time
            t = time.time()
            if quit_time is None or quit_time + 0.25 < time.time():
                quit_count = 0

            quit_time = t
            quit_count += 1
            if quit_count != 2:
                return
        root.destroy()


    @bind(root, "a")
    def add(event):
        pass


    def shrink_to_fit_idx(idx):
        if idx is None: return
        clip_rect = tuple(page_viewer.rects[idx])

        SCALE = 8 # 8
        COORD_ALIGN = 1 # SCALE

        pixel_perfect = False
        if pixel_perfect:

            org_pdf, page_index, parent_clip = document.pdf_page(dpage.page_number, (0,0,1,1))

            clip = translate_like(clip_rect, src=(0,0,1,1), dst=parent_clip)

            a_clip = pixel_align_rect(org_pdf.get_page_size(page_index), COORD_ALIGN, clip)
            #clip_rect = translate_like(a_clip, src=parent_clip, dst=(0,0,1,1))
            img = render_page_clip(org_pdf.get_page(page_index), None, None, a_clip, scale=SCALE)

            fit_rect = get_minimal_bounds(img) # TODO
            fit_rect = rect_to_parent(a_clip, fit_rect)

            fit_rect = pixel_align_rect(org_pdf.get_page_size(page_index), COORD_ALIGN, fit_rect)
            fit_rect = translate_like(fit_rect, src=parent_clip, dst=(0,0,1,1))


        else:
            img = page_viewer.image_func(None, None, clip_rect, scale=SCALE)

            fit_rect = get_minimal_bounds(img) # TODO

            fit_rect = rect_to_parent(clip_rect, fit_rect)

        page_viewer.set_rect(idx, fit_rect)

        clip_pages[dpage.page_number].subclips[idx].rect = Rectangle(*fit_rect)


    @bind(root, "e")
    #@bind(root, "m")
    def shrink_to_fit(event): # wrap current selection
        # examine pixels and shrink bounding box
        # Implement by scaling image down to 1 pixel, in the perpendicular axis to examine
        # Do this by halving the image recursively and blending both halves onto each other (minimum)

        idx = page_viewer.selection
        shrink_to_fit_idx(idx)

    #@bind(root, "<Key>")
    #def on_key(event):
    #    print(event)


    @bind(root, "<Shift-ISO_Left_Tab>")
    @bind(root, "<Tab>")
    #@bind(root, "<Shift-Tab>")
    def tab(event):
        delta = -1 if event.state & SHIFT_STATE_MASK else +1
        sel_index = page_viewer.selection
        if sel_index is None:
            sel_index = 0 if delta == +1 else len(page_viewer.rects)-1
        else:
            sel_index += delta
        if not (0 <= sel_index < len(page_viewer.rects)):
            sel_index = None
        page_viewer.select_rect(sel_index)



    @bind(root, "<F5>")
    def refresh(event):
        document.force_reload()
        page_viewer.redraw_canvas()


    @bind(root, "d")
    @bind(root, "<Delete>")
    def delete(event):
        page_viewer.remove_rect(page_viewer.selection)


    @bind(root, "z")
    def toggle_frac(event):
        page_viewer.fraction = 1
        page_viewer.pos_offset = (0, 0)
        page_viewer.redraw_canvas()

    @bind(root, "+")
    def toggle_frac(event):
        page_viewer.fraction *= 2**0.5
        page_viewer.redraw_canvas()

    @bind(root, "-")
    def toggle_frac(event):
        page_viewer.fraction /= 2**0.5
        page_viewer.redraw_canvas()

    @bind(root, "v")
    def copy_from_previous(event):
        if dpage.page_number == 0: return
        for c in clip_pages[dpage.page_number-1].subclips:
            copy = Clip(c.rect, [], c.style)
            clip_pages[dpage.page_number].subclips.append(copy)
        update_viewer_rects()


    def delta_for(event):
        return 10 if event.state & CONTROL_STATE_MASK else 1

    #@bind(root, "h")
    #@bind(root, "k")
    @bind(root, "<Up>")
    @bind(root, "<Left>")
    @bind(root, "<Button-4>")
    def prev(event):
        set_page(delta=-delta_for(event))

    #@bind(root, "j")
    #@bind(root, "l")
    @bind(root, "<space>")
    @bind(root, "<Return>")
    @bind(root, "<Down>")
    @bind(root, "<Right>")
    @bind(root, "<Button-5>")
    def next(event):
        set_page(delta=+delta_for(event))

    def pan(dx, dy, delta, to_border: bool):

        x, y = page_viewer.pos_offset
        if to_border:
            x *= not dx
            y *= not dy

            rx0, ry0, rx1, ry1 = page_viewer.image_rect
            w, h = page_viewer.canvas.winfo_width(), page_viewer.canvas.winfo_height()

            px = (rx1-rx0 - w)//2 * dx
            py = (ry1-ry0 - h)//2 * dy
            page_viewer.pos_offset = (x-px, y-py)
        else:
            page_viewer.pos_offset = (x-delta*dx, y-delta*dy)
        page_viewer.redraw_canvas()


    def handle_pan(event, dx, dy):
        delta = 160 if event.state & ALT_STATE_MASK else 40
        to_border = event.state & CONTROL_STATE_MASK
        pan(dx, dy, delta, to_border)

    @bind(root, "h")
    def pan_left(event): handle_pan(event, -1, 0)
    @bind(root, "j")
    def pan_down(event): handle_pan(event, 0, 1)
    @bind(root, "k")
    def pan_up(event): handle_pan(event, 0, -1)
    @bind(root, "l")
    def pan_right(event): handle_pan(event, 1, 0)

    @bind(root, "n")
    def next_page(event):
        dy = -1
        if not set_page(delta=+1):
            dy = +1
        page_viewer.pos_offset = (0,0)
        pan(0, dy, 1, to_border=True)

    @bind(root, "p")
    def prev_page(event):
        dy = +1
        if not set_page(delta=-1):
            dy = -1
        page_viewer.pos_offset = (0,0)
        pan(0, dy, 1, to_border=True)

    @bind(root, "w")
    #@bind(root, "<Control>s")
    def save_all(event):
        r = clip_pages.root
        d = r.asdict()
        fn = clip_file(r.filename)
        print(f"Writing to {fn}")
        with open(fn, "wt") as f:
            json.dump(d, f)

    @bind(root, "y")
    def sort_by_y(event):
        # TODO: recursively/all pages
        print("Sorting elements")
        clip_pages[dpage.page_number].subclips.sort(key=lambda clip: clip.rect.y0)
        update_viewer_rects()

    @bind_all(root, "<MouseWheel>")
    def scroll(event):
        set_page(delta=-event.delta/120)




    def view_current(rect_index, view_all : bool):
        if rect_index is None: rect_index = 0

        if view_all:
            clip_container, offsets = make_flattened_clip_container(clip_pages)

            rect_index += offsets[dpage.page_number]
            sub_document = FlattenedMultiSubpageDocument(
                document=document,
                page_number_offsets=offsets,
                clip_container=clip_container,
            )
        else:
            clip_container = clip_pages[dpage.page_number]
            sub_document = MultiSubpageDocument(
                document=document,
                page_number=dpage.page_number,
                clip_container=clip_container,
            )

        subclip_pages = MultiClipContainerList(
            root=clip_pages.root,
            clip_container=clip_container,
        )

        if len(clip_container.subclips) == 0: return

        rect_index = min(rect_index, len(clip_container.subclips)-1)

        show_image(
            name=str_title(False),
            document=sub_document,
            clip_pages=subclip_pages,
            visual_effect_index=visual_effect_index,
            page_number=rect_index
        )


    @bind(root, "<FocusIn>")
    def focus_in(event):
        if AUTO_RELOAD and dpage.needs_redraw():
            page_viewer.redraw_canvas()

    @bind(root, "<F2>")
    @bind(root, "i")
    def view(event):
        view_current(rect_index=page_viewer.selection, view_all=bool(event.state & CONTROL_STATE_MASK))

    @partial(setattr, page_viewer, "on_rect_click")
    def viewer_on_rect_click(event, index, rect):
        view_current(rect_index=index, view_all=bool(event.state & CONTROL_STATE_MASK))

    @bind(root, "<F3>")
    def find_similar(event):
        backwards = event.state & SHIFT_STATE_MASK
        if page_viewer.selection is not None:
            next_index = page_viewer.selection + (-1 if backwards else +1)
        else:
            next_index = None

        found_page = None
        found_clip = None

        if page_viewer.selection is not None and not (event.state & CONTROL_STATE_MASK):
            style = clip_pages[dpage.page_number].subclips[page_viewer.selection].style
            clip_pred = lambda clip: clip.style == style or ((clip.style and style) and (clip.style.mark_color == style.mark_color))
        else:
            clip_pred = lambda clip: True

        if not backwards:
            for pg in range(dpage.page_number, document.page_count):
                page = clip_pages[pg]
                start = 0
                if pg == dpage.page_number and next_index is not None:
                    start = next_index
                for i in range(start, len(page.subclips)):
                    if clip_pred(page.subclips[i]):
                        found_page = pg
                        found_clip = i
                        break
                else: continue
                break
        else:
            for pg in range(dpage.page_number, -1, -1):
                page = clip_pages[pg]
                start = len(page.subclips)-1
                if pg == dpage.page_number and next_index is not None:
                    start = next_index
                for i in range(start, -1, -1):
                    if clip_pred(page.subclips[i]):
                        found_page = pg
                        found_clip = i
                        break
                else: continue
                break


        if found_clip is not None:
            set_page(index=found_page)
            page_viewer.select_rect(found_clip)


    fullscreen = False
    @bind(root, "<F11>")
    def toggle_fullscreen(event):
        nonlocal fullscreen
        fullscreen ^= True
        root.attributes("-fullscreen", fullscreen)

    @bind(root, "<Escape>")
    def toggle_fullscreen(event):
        nonlocal fullscreen
        fullscreen = False
        root.attributes("-fullscreen", fullscreen)

    @bind(root, ":")
    def toggle_fullscreen(event):
        print("colon")


    def make_clip_directory():
        dir_filename = f"{clip_pages.root.filename}-clips"
        os.makedirs(dir_filename, exist_ok=True)
        return dir_filename

    def make_clip_filename(dir_filename, page_index, clip, ext):
        clip = [int(c*99) for c in clip]
        l,t,r,b=clip

        clip_base_filename = f"{page_index}-{l}-{t}-{r}-{b}.{ext}"
        return os.path.join(dir_filename, clip_base_filename)

    @bind(root, "x")
    def extract_selection(event):
        if page_viewer.selection is None:
            clip_rect = (0,0,1,1)
        else:
            clip_rect = tuple(clip_pages[dpage.page_number].subclips[page_viewer.selection].rect)

        doc = pdfium.PdfDocument.new()
        org_pdf, page_index, clip = document.pdf_page(dpage.page_number, clip_rect)
        page_size = org_pdf.get_page_size(page_index)
        page_rect = (0, 0, *page_size)
        page_clip = rect_to_parent(page_rect, clip)
        clip_size = page_clip[2]-page_clip[0], page_clip[3]-page_clip[1]
        new_page = doc.new_page(*page_size)
        xpage = org_pdf.page_as_xobject(page_index, doc)
        new_page.insert_object(xpage.as_pageobject())
        new_page.set_mediabox(page_clip[0], page_size[1]-page_clip[3], page_clip[2], page_size[1]-page_clip[1])
        #new_page.set_cropbox(page_clip[0], page_size[1]-page_clip[3], page_clip[2], page_size[1]-page_clip[1])
        #new_page.set_trimbox(page_clip[0], page_size[1]-page_clip[3], page_clip[2], page_size[1]-page_clip[1])
        new_page.generate_content() # completed

        # TODO

        #remove_list = []
        #for obj in new_page.get_objects(max_depth=30):
        #    print(obj, obj.type, obj.get_pos())
        #    try: print(obj.get_info())
        #    except: pass
        #    l,b,r,t = obj.get_pos()
        #    r = page_size[0]-r
        #    b = page_size[1]-b
        #    if l > page_clip[2] or t > page_clip[3] \
        #    or r < page_clip[0] or b < page_clip[1]:
        #        remove_list.append(obj)
        #
        #for obj in reversed(remove_list):
        #    pdfium.FPDFPage_RemoveObject(new_page.raw, obj.raw)
        #
        #new_page.generate_content() # completed

        CAIRO_ARGS = []

        dir_filename = make_clip_directory()
        clip_filename = make_clip_filename(dir_filename, page_index, clip, "pdf")

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
            #efn = extract_filename(clip_pages.root.filename)
            msg = f"Extracting to {clip_filename} using {tf.name}"
            print(msg)
            with open(tf.name, "wb") as f:
                doc.save(f)

            #run_exec(["pdfcrop", tf.name, clip_filename])
            run_exec(["pdftocairo", "-pdf", *CAIRO_ARGS, tf.name, clip_filename])

            run_exec(["which", "pdftex"])

        #subprocess.call(["pdf2dsc", clip_filename, clip_filename+".dsc"])
        #subprocess.call(["pdf2ps", "-dLanguageLevel=3", clip_filename, clip_filename+".eps"])
        #subprocess.call(["ps2pdf", clip_filename+".eps", clip_filename+".pdf"])
        #subprocess.call(["pdftops", "-level3", "-eps", clip_filename, clip_filename+".eps"])
        #subprocess.call(["pstoedit", "-f", "plot-svg", clip_filename+".eps", clip_filename+".svg"])

        #svg_filename = os.path.splitext(clip_filename)[0]+".svg"

        #svg_filename = clip_filename+".svg"
        #run_exec(["pdftocairo", "-svg", *CAIRO_ARGS, clip_filename, svg_filename])
        #copyname = svg_filename
        png_filename = clip_filename+".png"
        run_exec(["pdftocairo", *CAIRO_ARGS, "-singlefile", "-png", clip_filename, png_filename[:-4]])

        if event.state & CONTROL_STATE_MASK:
            img = Image.open(png_filename)
            def ceildivi(a, d): return int((a+d-1)/d)
            img = img.resize((ceildivi(img.width,1.5), ceildivi(img.height,1.5)), resample=Image.LANCZOS)
            img = page_viewer.visual_effect_func(img)
            img.save(png_filename)
        copyname = png_filename

        rel_name = os.path.join(os.path.basename(dir_filename), os.path.basename(copyname))
        #rel_name = rel_name.replace(' ', "%20")

        import urllib.parse
        rel_name = urllib.parse.quote(rel_name)

        root.clipboard_clear()
        root.clipboard_append(f"![]({rel_name})")



    def copy_selection_as_texinclude():
        if page_viewer.selection is None:
            clip_rect = (0,0,1,1)
        else:
            clip_rect = tuple(clip_pages[dpage.page_number].subclips[page_viewer.selection].rect)

        org_pdf, page_index, clip = document.pdf_page(dpage.page_number, clip_rect)

        l,t,r,b=clip
        r=1-r
        b=1-b

        _, fn = os.path.split(clip_pages.root.filename)
        tex = f"\\adjincludegraphics[page={page_index+1},trim={{{{{l:.5}\width}} {{{b:.5}\height}} {{{r:.5}\width}} {{{t:.5}\height}}}},clip]{{{fn}}}"
        root.clipboard_clear()
        root.clipboard_append(tex)

    def copy_selection_as_png_md():
        if page_viewer.selection is None:
            clip_rect = (0,0,1,1)
        else:
            clip_rect = tuple(clip_pages[dpage.page_number].subclips[page_viewer.selection].rect)

        img = page_viewer.image_func(None, None, clip_rect, scale=2)


        org_pdf, page_index, clip = document.pdf_page(dpage.page_number, clip_rect)

        dir_filename = make_clip_directory()
        clip_filename = make_clip_filename(dir_filename, page_index, clip, "png")

        img.save(clip_filename)
        root.clipboard_clear()
        rel_name = os.path.join(os.path.basename(dir_filename), os.path.basename(clip_filename))
        root.clipboard_append(f"![]({rel_name})")

    @bind(root, "c")
    def copy_selection(event):
        if event.state & SHIFT_STATE_MASK:
            copy_selection_as_texinclude()
        else:
            copy_selection_as_png_md()

    root.mainloop()



def show_pdf(name, filename):
    pdf_doc = PDFDocument(filename)

    clip_doc_obj = {"filename":filename, "pages":[]}
    if os.path.exists(clip_file(filename)):
        with open(clip_file(filename), "rt") as f:
            clip_doc_obj = json.load(f)
            clip_doc_obj["filename"] = filename
        print("JSON loaded from", clip_file(filename))

    clip_doc = ClipDocument.fromdict(clip_doc_obj)
    #print(clip_doc)

    show_image(name, document=PDFDocument(filename), clip_pages=clip_doc)





if __name__=="__main__":
    argv = sys.argv

    if len(argv) != 2:
        print("Usage: view <file.pdf>")
        exit(1)

    filename = argv[1]
    try:
        show_pdf(os.path.basename(filename), filename)
    except:
        import traceback
        print("ERROR OCCURRED")
        traceback.print_exc()
