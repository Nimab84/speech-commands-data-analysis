"""Shared English report layout for the Phase 2 measurements."""
import hashlib
import json
import math
import re
from html import escape
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/tables'
BUILD = ROOT / 'tmp/report'
FIG = ROOT / 'figures'
W, H = A4
M = 45
CW = W - 2*M
NAVY = HexColor('#16364d')
TEAL = HexColor('#167d8d')
INK = HexColor('#263849')
MUTED = HexColor('#657686')
LIGHT = HexColor('#eff4f7')
BORDER = HexColor('#d7e1e8')
AUTHOR = 'Nima Balotf'

EMPHASIS = (
    'Question 1', 'Macro-F1',
    'Balanced commands do not imply balanced speaker contributions',
    'most frequent training class', 'speaker-level bootstrap',
)

FIGURE_NOTES = {
    'class_split_counts': 'Command counts by speaker partition',
    'train_distributions': 'Training distributions: duration, energy, and peak amplitude',
    'train_class_relationships': 'Within-class short-clip rates and amplitude summaries in train',
    'train_multivariate': 'Training relationships and rank correlations',
    'speaker_coverage': 'Corpus-wide contribution and command coverage per ID',
}


def wrap(text, width, font, size):
    lines, line = [], ''
    for word in str(text).split():
        candidate = f'{line} {word}'.strip()
        if line and pdfmetrics.stringWidth(candidate, font, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    if any(pdfmetrics.stringWidth(line, font, size) > width for line in lines):
        raise ValueError(f'Unbreakable text exceeds its column: {text}')
    return lines


class Report:
    def __init__(self, path, total_pages):
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.c.setTitle('Phase 2 - Audio Features and Unsupervised Patterns')
        self.c.setAuthor(AUTHOR)
        self.c.setSubject('Speech Commands mini: preparation, features, PCA, clustering and stability')
        self.page = 0
        self.y = 0
        self.pages = []
        self.page_bottoms = []
        self.figure_number = 0
        self.total_pages = total_pages

    def new(self, title, label='Preparation, audio features, and unsupervised patterns'):
        if self.page:
            self.finish()
        self.page += 1
        self.pages.append(title)
        c = self.c
        c.setFillColor(NAVY)
        c.rect(M, H-39, 24, 4, stroke=0, fill=1)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(M+34, H-40, 'PATH D / PHASE 02')
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 8)
        c.drawRightString(W-M, H-40, 'GOOGLE SPEECH COMMANDS MINI')
        c.setStrokeColor(BORDER)
        c.line(M, H-52, W-M, H-52)
        if self.page == 1:
            c.setFillColor(NAVY)
            c.setFont('Helvetica-Bold', 33)
            c.drawString(M, H-110, 'Audio features')
            c.drawString(M, H-149, 'and hidden patterns')
            # Original vector motif: decorative rhythm, not a measured audio plot.
            c.setStrokeColor(TEAL)
            c.setLineWidth(2.4)
            for i in range(25):
                x = W-M-111+i*4.4
                height = 7+30*math.exp(-((i-12)/6)**2)*abs(math.cos(i*1.15))
                c.line(x,H-118-height/2,x,H-118+height/2)
            self.y = H-181
            self.p(label,size=9,color=MUTED,gap=14,leading=14)
            self.p(AUTHOR,size=11,color=NAVY,gap=14,leading=15)
            return
        self.y = H-86
        size = 20
        while pdfmetrics.stringWidth(title, 'Helvetica-Bold', size) > CW:
            size -= .25
        c.setFont('Helvetica-Bold', size)
        c.setFillColor(NAVY)
        c.drawString(M, self.y, title)
        self.y -= 16
        c.setStrokeColor(TEAL)
        c.setLineWidth(2)
        c.line(M,self.y,M+34,self.y)
        c.setLineWidth(.5)
        self.y -= 19
        if label != 'Preparation, audio features, and unsupervised patterns':
            self.p(label,size=8.7,color=MUTED,gap=10,leading=14)

    def finish(self):
        if self.y < 54:
            raise ValueError(f'Page {self.page} overflow: {self.y}')
        self.page_bottoms.append(round(self.y, 2))
        c = self.c
        c.setStrokeColor(BORDER)
        c.setLineWidth(.5)
        c.line(M, 43, W-M, 43)
        c.setFont('Helvetica-Bold', 8)
        c.setFillColor(NAVY)
        c.drawString(M, 28, 'PHASE 2 / FEATURES AND PATTERNS')
        c.setFont('Helvetica', 8)
        c.setFillColor(MUTED)
        c.drawRightString(W-M-43, 28, '04 NOV 2026')
        c.setFillColor(NAVY)
        c.roundRect(W-M-31,20,31,19,3,stroke=0,fill=1)
        c.setFillColor(white)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(W-M-15.5,26,f'{self.page:02d}/{self.total_pages:02d}')
        c.showPage()

    def p(self, text, size=10.3, color=INK, gap=9, leading=16):
        markup = escape(str(text))
        if '**' not in markup:
            for phrase in EMPHASIS:
                markup = markup.replace(escape(phrase),'<b>'+escape(phrase)+'</b>')
        markup = re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',markup)
        style = ParagraphStyle('Body',fontName='Helvetica',fontSize=size,
                               leading=leading,textColor=color,spaceAfter=0)
        paragraph = Paragraph(markup,style)
        _,height = paragraph.wrap(CW,H)
        paragraph.drawOn(self.c,M,self.y-height+size)
        self.y -= height+gap

    def section(self, text):
        self.y -= 4
        self.c.setFillColor(TEAL)
        self.c.rect(M,self.y-1,3,11,stroke=0,fill=1)
        self.c.setFont('Helvetica-Bold', 12)
        self.c.setFillColor(NAVY)
        self.c.drawString(M+12, self.y, text)
        self.y -= 24

    def analysis(self, observation, interpretation, next_step, size=9.8, leading=15):
        for label, text in [('Observation', observation), ('Interpretation', interpretation),
                            ('Next step', next_step)]:
            self.p(f'**{label}.** {text}', size=size, leading=leading, gap=7)

    def latin(self, text, size=8.5, gap=14):
        self.c.setFillColor(MUTED)
        self.c.setFont('Helvetica', size)
        if pdfmetrics.stringWidth(str(text), 'Helvetica', size) > CW:
            raise ValueError(f'Unwrapped line: {text}')
        self.c.drawString(M, self.y, str(text))
        self.y -= gap

    def link(self, label, url, size=8.5):
        self.c.setFillColor(TEAL)
        self.c.setFont('Helvetica', size)
        self.c.drawString(M, self.y, label)
        self.c.linkURL(url, (M, self.y-2, M+pdfmetrics.stringWidth(label,'Helvetica',size), self.y+size), relative=0)
        self.y -= 18

    def kpis(self, items):
        gap = 12
        width = (CW-gap)/2
        top = self.y
        for i,(value,label) in enumerate(items):
            x = M+(i%2)*(width+gap)
            y = top-(i//2)*68
            self.c.setFillColor(LIGHT)
            self.c.roundRect(x,y-57,width,57,4,stroke=0,fill=1)
            self.c.setFillColor(TEAL)
            self.c.rect(x+12,y-12,22,2,stroke=0,fill=1)
            self.c.setFillColor(NAVY)
            self.c.setFont('Helvetica-Bold',23)
            self.c.drawString(x+12,y-36,str(value))
            self.c.setFillColor(MUTED)
            self.c.setFont('Helvetica',8.5)
            self.c.drawString(x+12,y-49,label)
        self.y = top-136

    def callout(self, title, text):
        style = ParagraphStyle('Note',fontName='Helvetica',fontSize=9.2,leading=14,textColor=INK)
        paragraph = Paragraph(escape(text),style)
        _,height = paragraph.wrap(CW-24,H)
        total = height+37
        self.c.setFillColor(LIGHT)
        self.c.roundRect(M,self.y-total,CW,total,4,stroke=0,fill=1)
        self.c.setFillColor(TEAL)
        self.c.rect(M,self.y-total,3,total,stroke=0,fill=1)
        self.c.setFont('Helvetica-Bold',9.2)
        self.c.setFillColor(NAVY)
        self.c.drawString(M+12,self.y-16,title)
        paragraph.drawOn(self.c,M+12,self.y-25-height)
        self.y -= total+14

    def table(self, headers, rows, widths=None, row_h=25, size=9):
        c = self.c
        widths = widths or [CW/len(headers)]*len(headers)
        for n, cells in enumerate([headers] + list(rows)):
            font = 'Helvetica-Bold' if n == 0 else 'Helvetica'
            content = [wrap(cell, width-16, font, size) for cell,width in zip(cells,widths)]
            height = max(row_h, max(map(len,content))*12+12)
            c.setFillColor(NAVY if n == 0 else LIGHT if n % 2 else white)
            c.rect(M, self.y-height+6, CW, height, stroke=0, fill=1)
            x = M
            for cell, lines, width in zip(cells,content,widths):
                c.setFont(font, size)
                c.setFillColor(white if n == 0 else INK)
                top = self.y-(height-len(lines)*12)/2-4
                numeric = bool(re.fullmatch(r'[0-9.,+% /-]+',str(cell)))
                if numeric:
                    c.setFont('Helvetica-Bold',size)
                for i,line in enumerate(lines):
                    if numeric:
                        c.drawCentredString(x+width/2, top-i*12, line)
                    else:
                        c.drawString(x+8, top-i*12, line)
                x += width
            self.y -= height
        self.y -= 16

    def figure(self, name, width=CW, max_h=None):
        path = FIG / f'{name}.png'
        with PILImage.open(path) as image:
            iw,ih = image.size
        height = width*ih/iw
        if max_h and height > max_h:
            height = max_h
            width = height*iw/ih
        self.c.drawImage(str(path), M+(CW-width)/2, self.y-height,
                         width=width, height=height, preserveAspectRatio=True, mask='auto')
        self.y -= height+10
        self.figure_number += 1
        note = FIGURE_NOTES.get(name, f'{name.removeprefix("examples_").upper()}: three fixed training examples')
        self.p(f'**FIGURE {self.figure_number:02d}** / {note}',size=8,color=MUTED,leading=11,gap=10)

    def save(self):
        if self.page != self.total_pages:
            raise ValueError(f'Expected {self.total_pages} pages, found {self.page}')
        self.finish()
        self.c.save()
