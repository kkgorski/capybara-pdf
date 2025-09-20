#!/usr/bin/env python3

import sys
import fitz
import easyocr
import statistics
import numpy as np
import logging
from pprint import pprint
import cv2 as cv
# sudo apt install libgtk2.0-dev pkg-config

from lxml import etree


class TextEntry:
    def __calc_line_index(self, y, y_max):
        y_percentage = 1 - ((y_max - y) / y_max) # percentage is inverted, because 0th index is topmost line`
        return round(y_percentage * num_of_lines)

    def __calc_char_index(self, x, x_max):
        x_percentage = 1 - ((x_max - x) / x_max) # percentage is inverted, because 0th index is leftmost char
        return round(x_percentage * num_of_chars)


    def __init__(self, bbox, text):
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = bbox

        xs = [i[0] for i in bbox]
        ys = [i[1] for i in bbox]

        #  BBOX definition
        #
        #  {x3, y3}                    {x2, y2}
        #  +---------------------------+
        #  |                           |
        #  # <- (y_avg, x_min)         |
        #  |                           |
        #  +---------------------------+
        #  {x0, y0}                    {x1, y1}
        #

        x_min = min(xs)
        y_avg = sum(ys) / len(ys)

        x_char = self.__calc_char_index(x_min, pix.width)
        y_char = self.__calc_line_index(y_avg, pix.height)

        # print(f'{text:<50}| {{xc,yc}}={{{x_char:>3},{y_char:>3}}}, {{x,y}}={{{x_min:>5},{y_avg:>5}}}')
        print(f'line: {y_char:>3} char: {x_char:>3} {{x,y}}={{{x_min:>5},{y_avg:>5}}} | {text:<60}')

        self.text = text
        # x coordinate as number of characters
        self.x_min = x_char
        self.x_max = x_char + len(text)
        # x coordinate as pixels
        self.x_pix_min = x_min
        # y coordinate as number of characters
        self.y_avg = y_avg
        self.y_char = y_char


class SpecPage:
    def __init__(self):
        self.lines = {}

    def add(self, bbox, text):
        text = TextEntry(bbox, text)
        y_char = text.y_char

        if y_char not in self.lines:
            self.lines[y_char] = [text]
        else:
            prev_text = self.lines[y_char][-1]
            if prev_text.x_max >= text.x_min:
                logging.warning(f"Bounding boxes overlap {prev_text.x_max} >= {text.x_min} for \"{prev_text.text}\" and \"{text.text}\"\n" \
                              "Please increase number of chars")
            self.lines[y_char].append(text)

    def print(self, report):

        prev_y = 0
        # TODO add first pass where everything is shifted to the left as much as possible. Maybe we can do that somewhere else?
        # TODO add validating pass where x_max is checked against the next x_min. Don't know how to deal with that yet...
        #      ideas -> padding entry for every line in the page
        #            -> rerunning the program with increased line length - I think this should be one of the config params
        for y, texts in self.lines.items():
            line = ""
            for text in texts:
                while len(line) < text.x_min:
                    line += " "
                line += text.text

            print(line, file=report, end='')
            newlines = y - prev_y
            [print(file=report) for _ in range(newlines)]
            prev_y = y



class Spec():
    def __init__(self):
        self.pages = {}

    def add(self, num, page):
        self.pages[num] = page

    def print(self):
        with open('out.txt', 'w') as report:
            for i, page in self.pages.items():
                page.print(report)

    def debug_print_ocr(self, ocr):
        detected_data = []
        for i, detected_text in enumerate(ocr):
            (bbox, text, accuracy) = detected_text

            printable_bbox = []
            for items in bbox:
                printable_items = []
                for item in items:
                    printable_items.append(int(item))
                printable_bbox.append(printable_items) # to convert from numpy to regular int

            detected_data.append((i, printable_bbox, text, accuracy))

        for i, printable_bbox, text, accuracy in detected_data:
            print(i, text, accuracy, printable_bbox)

    def process_page_ocr(self, ocr):
        spec_page = SpecPage()
        for detected_text in ocr:
            (bbox, text, accuracy) = detected_text
            spec_page.add(bbox, text)
        spec.add(page.number, spec_page)

        self.debug_print_ocr(ocr)

    def find_tables(self, keywords):
        for i, page in self.pages.items():
            table_found = False
            #x_offsets = None
            matching_keywords = {}
            data = [ [] for keyword in keywords ]
            for y, texts in page.lines.items():

                if not table_found:
                    for text in texts:
                        for keyword in keywords:
                            print(keyword, text.text)
                            if keyword in text.text:
                                print("KUBA Match!!!")
                                matching_keywords[keyword] = text.x_pix_min

                    if len(matching_keywords) > 2:
                        print(f"=================================================== {matching_keywords}")
                        len_texts = len(texts)
                        print(f"len texts {len_texts}")

                    if len(matching_keywords) == len(keywords):
                        table_found = True
                        print("============================================================================ OK !!! ")
                        print(f"Table found, line {y}")
                        #x_offsets = { keyword : text.x_pix_min for text in texts }
                        #print(x_offsets)

                    if not table_found:
                        matching_keywords = {} # must be found in the same line

                if table_found:

                    missing_matches = { keyword : i for i, keyword in enumerate(keywords) }
                    # set(range(len(matching_keywords.items())))

                    print(f'=== Line {y} ============')
                    for text in texts:
                        for i, (keyword, x_pix_min) in enumerate(matching_keywords.items()):
                            my_diff = abs(x_pix_min - text.x_pix_min) < 15.0  # this should be configurable
                            if my_diff:                                       # ideally calculated from width / height
                                print(f'{keyword}={text.text}, idx {i}')
                                data[i].append(text.text)
                                try:
                                    missing_matches.pop(keyword)
                                except KeyError:
                                    pass # We have seen stuff at this index, maybe we should append?

                    if missing_matches:
                        print("Missing matches: ", missing_matches)
                    print()

                    for i in missing_matches.values():
                        data[i].append("") # Add empty entries when there was no match

                    # TODO
                    # create a list of lists padded with "" to represent a table

                        #if text.text == "MatchingUnit":
                        #    print(x_pix_min)
                        #    print(f'{keyword}, {text.text}, {text.x_pix_min}, {my_diff}')

            pprint(data)

class OcrCreator:
    def __init__(self, imgfile, scale_factor = 2):
        self.image = cv.imread(imgfile) # #TODO IMPORTANT! this thing we changed
        print(pix.width, pix.height)
        self.scale_factor = scale_factor # TODO from config

    def save_debug(self, name, img, ocr):
        img_annotated = img.copy()

        for elem in ocr:
            p00 = tuple(map(int, elem[0][0]))
            p01 = tuple(map(int, elem[0][1]))
            p02 = tuple(map(int, elem[0][2]))
            p03 = tuple(map(int, elem[0][3]))
            color = (0, 255, 0)
            img_annotated = cv.line(img_annotated, p00, p01, color, 2)
            img_annotated = cv.line(img_annotated, p01, p02, color, 2)
            img_annotated = cv.line(img_annotated, p02, p03, color, 2)
            img_annotated = cv.line(img_annotated, p03, p00, color, 2)

        cv.imwrite(name, img_annotated)

    def preprocess(self, img):
        upscaled_img = cv.resize(img, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv.INTER_LINEAR)
        blurred_img = cv.blur(upscaled_img, (5, 5))
        return blurred_img
        # https://github.com/JaidedAI/EasyOCR/issues/341

    def do_ocr(self, img, debug_filename = None):
        preprocessed_img = self.preprocess(img)
        ocr = reader.readtext(preprocessed_img, min_size=1, text_threshold=0.05, canvas_size=3300) # needs adjusting to find single '1'
        if debug_filename:
            self.save_debug(debug_filename, preprocessed_img, ocr)
        return ocr

    def ocr_image(self):

        # TODO
        # implement detection algoritm to check if we're not crossing anything while
        # cutting the page in half. Don't know how likely is that - leaving it for later

        # We create ocr 2 times for upper and lower half
        # to fit withing 8GB ram limit

        vertical_half = int(pix.height / 2)
        x0, x1 = ( 0, pix.width )
        y0, y1 = ( 0, vertical_half )
        y1, y2 = ( vertical_half, pix.height )

        upper_half_of_img = self.image[y0:y1, x0:x1]
        lower_half_of_img = self.image[y1:y2, x0:x1]

        upper_ocr = self.do_ocr(upper_half_of_img, "upper.png")
        lower_ocr = self.do_ocr(lower_half_of_img, "lower.png")

        ocr = upper_ocr

        for bbox, text, accuracy in lower_ocr:
            # [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = bbox

            # we need to adjust for scaling factor here, so we shift by half image verticaly
            shifted_box = [(x, y + (self.scale_factor * vertical_half)) for x, y in bbox]
            ocr.append([shifted_box, text, accuracy])

        original_size_ocr = []
        for [bbox, text, accuracy] in ocr:
           # ocr was done for scaled image
           original_bbox = [ (x/self.scale_factor, y/self.scale_factor) for x, y in bbox ]
           original_size_ocr.append([original_bbox, text, accuracy])

        self.save_debug("full.png", self.image, original_size_ocr)

        return ocr

# Spec specific
num_of_lines = 46
num_of_chars = 140
table_keywords = ["FIELD", "OFFSET", "LENGTH", "DATA TYPE", "DESCRIPTION"]
# OCR
dpi = 100
scale_factor = 2
# Debug options
page_filer = [26]
# Logging

spec = Spec()
reader = easyocr.Reader(['en'], gpu=False)

with fitz.open(sys.argv[1]) as doc:
    for page in doc:
        if page.number not in page_filer:
            continue
        pix = page.get_pixmap(dpi=dpi)
        filename = f"page{page.number}.png"
        pix.save(filename)

        ocr_creator = OcrCreator(filename, scale_factor)
        ocr_result = ocr_creator.ocr_image()
        spec.process_page_ocr(ocr_result)

spec.print()
spec.find_tables(table_keywords)

# TODO
# Type = {REF, BUG, FEA} = {Refactoring, Bug fix, New Feature}
# 
# R | TYP | DESCRIPTION
# --+-----+---------------
# 1 | FEA | Improve logging
# 1 | FEA | To xml conversion
# 1 | FEA | Add help
# 0 | FEA | Json config
# 0 | FEA | Implement OCR cache to skip the ocr process.
# 0 | FEA | Non ocr version <PROBABLY IMPORTANT>
# 0 | FEA | Text cut in half detection
# 0 | FEA | Add margins to the config
# 0 | FEA | Add argparser
#
# DONE
# 1 | REF | Create git repository
# 1 | REF | Move 'Spec page' code to 'Spec' class
