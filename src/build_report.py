"""Build the English Phase 1 PDF from the exported measurements."""
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
OUT = ROOT / 'reports/phase1'
BUILD = ROOT / 'tmp/phase1_build'
FIG = ROOT / 'figures/phase1'
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
        self.c.setTitle('Phase 1 - Audio Commands for Unseen Speakers')
        self.c.setAuthor(AUTHOR)
        self.c.setSubject('Speech Commands mini: quality, EDA, and speaker-disjoint evaluation')
        self.page = 0
        self.y = 0
        self.pages = []
        self.page_bottoms = []
        self.figure_number = 0
        self.total_pages = total_pages

    def new(self, title, label='Problem understanding, data quality, and exploratory analysis'):
        if self.page:
            self.finish()
        self.page += 1
        self.pages.append(title)
        c = self.c
        c.setFillColor(NAVY)
        c.rect(M, H-39, 24, 4, stroke=0, fill=1)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(M+34, H-40, 'PATH D / PHASE 01')
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 8)
        c.drawRightString(W-M, H-40, 'GOOGLE SPEECH COMMANDS MINI')
        c.setStrokeColor(BORDER)
        c.line(M, H-52, W-M, H-52)
        if self.page == 1:
            c.setFillColor(NAVY)
            c.setFont('Helvetica-Bold', 33)
            c.drawString(M, H-110, 'Audio commands')
            c.drawString(M, H-149, 'for unseen speakers')
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
        if label != 'Problem understanding, data quality, and exploratory analysis':
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
        c.drawString(M, 28, 'PHASE 1 / DATA AUDIT')
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


def main():
    s = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    q = s['quality']
    source = json.loads((ROOT/'data/source_manifest.json').read_text())
    cfg = json.loads((ROOT/'configs/phase1.json').read_text())
    counts = pd.read_csv(OUT/'class_split_counts.csv',index_col=0)
    stats = pd.read_csv(OUT/'numeric_summary_train.csv',index_col=0)
    inventory = pd.read_csv(OUT/'audio_inventory.csv')
    train = inventory[(inventory.status == 'ok') & (inventory.split == 'train')]
    BUILD.mkdir(parents=True,exist_ok=True)
    final_path = ROOT/'reports/phase1.pdf'
    build_path = BUILD/'phase1.pdf'
    r = Report(build_path, total_pages=13+len(cfg['labels']))
    pct = lambda x:f'{x*100:.1f}%'

    r.new('Audio Commands for Unseen Speakers', 'Phase 1 report | Path D: Google Speech Commands')
    r.p('This report examines the quality and structure of the Speech Commands mini dataset before feature extraction and modeling. It covers the audio files, speaker distribution, data split, and exploratory plots.',size=11,leading=18)
    r.section('Measured corpus snapshot')
    r.kpis([
        [f"{s['n_files']:,}",'Audited WAV clips'],
        [f"{s['n_speakers']:,}",'Speaker/session IDs'],
        [str(s['n_labels']),'Command classes'],
        [f"{s['total_duration_minutes']:.2f} min",'Total recorded duration'],
    ])
    r.callout('Integrity and partition checks',
              f"{q['corrupt_files']} unreadable files | {s['speaker_overlap_count']} shared IDs | "
              f"{s['cross_split_duplicate_groups']} exact PCM-copy groups crossing partitions")
    r.section('Main evaluation-design finding')
    r.p('All recordings from each speaker were assigned to one partition. The resulting training, validation, and test sets have no shared speaker IDs or exact audio copies. These sets will be used in the later modeling stages.')
    r.section('Scope of the conclusion')
    r.p('The task is to classify eight commands from short, pre-segmented recordings. Phase 1 focuses on data exploration and a constant baseline; audio classification and noise experiments follow in later phases.')
    r.latin('Evidence: summary.json / audio_inventory.csv / source_manifest.json',size=8)

    r.new('Problem Definition and Main Question')
    r.section('Application and supported decision')
    r.p('A simple voice interface must identify direction, movement, stopping, or confirmation. The technical decision is which compact representation and model can recognize commands from unseen speakers reliably. Evaluation must reflect that target population rather than repeated recordings from familiar IDs.')
    r.section('Three feasible questions')
    r.p('1. Can an audio-feature model outperform a constant baseline on commands from unseen speakers, and how does performance change on low-energy clips?')
    r.p('2. Do duration, energy, and low-energy-frame fractions differ by command, and could these variables become undesirable shortcuts?')
    r.p('3. Do clusters of spectral representations align more strongly with command, speaker/session, or energy?')
    r.section('Selected main question')
    r.p('Question 1 supports a practical decision, has measurable criteria, and allows classical and audio-model comparisons in later phases. Question 2 is diagnostic in Phase 1; Question 3 supplies the proposed unsupervised component for Phase 2.')
    r.table(['Component','Definition'],[
        ['Analysis/prediction unit','One short WAV utterance'],
        ['Split unit','Speaker/session filename prefix'],
        ['Target','One of the eight selected commands'],
        ['Target population','Similar English recordings from unseen IDs'],
        ['Proposed primary metric','Macro-F1']
    ],widths=[CW*.36,CW*.64])
    r.p('Phase 1 provides the data audit and evaluation setup for the main question. The later modeling phases will compare audio features and classifiers.',size=9.7)

    r.new('Provenance, Version, and Collection Limits')
    r.section('Exact identity of the analyzed data')
    r.p('The project uses every WAV file in the official mini archive. Its README describes an excerpt of Speech Commands and references v0.01 [2]. The full TFDS configuration has a different class structure [3].')
    r.latin('Archive: mini_speech_commands.zip | 182,082,353 bytes')
    r.latin('SHA-256: '+source['archive_sha256'],size=7.7)
    r.section('Collection process')
    r.p('Participants read prompted words through phone or laptop microphones. The collection combined volunteer recordings and paid crowdsourcing. Audio was converted to 16 kHz WAV and screened with automated and manual checks [1].')
    r.section('Time, observations, and identifiers')
    r.p('The first dataset version was released in 2017 and the paper appeared in 2018. Mini provides speaker IDs and recording counters in filenames, but no recording dates.')
    r.section('Bias and generalization')
    r.p('The dataset contains isolated English words with balanced class counts. A continuous voice interface will also need unrelated speech, background noise, and realistic command frequencies.')
    r.section('Data license')
    r.p('Attribute Speech Commands to Pete Warden / Google and mini packaging to TensorFlow. The parent data license is CC BY 4.0. IDs are used only for grouping and aggregate analysis, not personal identification.')
    r.latin('Sources [1]-[3] are listed with direct links in References.',size=8)

    r.new('Data Structure and Reproducible Flow')
    r.p('The raw input is a directory tree, not a ready-made table: each word folder contains WAV files. The audit produces one row per clip. Speaker-to-clip is one-to-many; speaker-to-command is many-to-many. Relative file paths form the primary key.')
    r.section('Data flow')
    for line in [
        'Official ZIP -> size + SHA-256 + CRC -> unchanged WAV files',
        'WAV header + PCM -> inventory -> quality flags + speaker/session ID',
        'Speaker hash -> fixed split -> train EDA + reproducible figures',
        'Exported CSV / JSON -> English PDF + consistency verification']:
        r.latin(line,size=8.8,gap=22)
    r.section('Key variable dictionary')
    r.table(['Field','Type','Role / unit'],[
        ['path','String','Unique clip key'],
        ['label','Category','Target from folder name'],
        ['speaker_id','String','Grouping ID, not a feature'],
        ['utterance_index','Integer','Counter, not a date'],
        ['split','Category','Fixed partition membership'],
        ['sample_rate / frames / channels','Integer','Hz, frame count, channel count'],
        ['duration_s','Float','Frames divided by sample rate'],
        ['rms_dbfs / peak_abs','Float','Energy and normalized amplitude'],
        ['file_sha256 / pcm_sha256','String','Exact-content fingerprints'],
        ['low_energy / very_short','Boolean','Audit rules, not new labels']
    ],widths=[CW*.42,CW*.14,CW*.44],size=8.5)
    r.p('Paths identify files, folder names supply labels, and speaker IDs define groups for the split. These fields are kept separate from the audio features used for prediction.',size=9.7)
    r.latin(f"Inventory schema: {len(inventory):,} rows x {len(inventory.columns)} columns")

    r.new('Quality Audit: Counts and Decisions')
    r.p('All WAV headers and samples were read. The thresholds below are initial, project-specific screening rules for review; they are not validated standards for corruption.')
    r.table(['Check / rule','Count','Initial decision'],[
        ['Read failure / truncated payload',str(q['corrupt_files']),'Quarantine if observed'],
        ['Rate not 16000 Hz or non-mono',str(q['unexpected_format']),'Review; standardize in Phase 2'],
        ['Duration below 0.25 s',str(q['very_short']),'Semantic review; no automatic removal'],
        ['Duration below 1 s',str(q['shorter_than_one_s']),'Retain; pad in later preprocessing'],
        ['Duration above 1 s',str(q['longer_than_one_s']),'Review before cropping'],
        ['RMS below -45 dBFS',str(q['low_energy']),'Review; do not label as silence'],
        ['Every sample equals zero',str(q['exact_silence']),'Inspect content'],
        ['At least one endpoint PCM value',str(q['saturated_files']),'Review possible saturation'],
        ['Absolute mean above 0.05',str(q['high_dc']),'Review DC offset'],
        ['Extra exact PCM copies',str(q['duplicate_extra_records']),'One fixed representative in Phase 2'],
        ['Missing/invalid speaker ID',str(q['missing_speaker_id']),'Stop split assignment if observed']
    ],widths=[CW*.43,CW*.10,CW*.47],row_h=27,size=8.5)
    r.section('Ranges, missing values, and copies')
    r.p(f"All files use {next(iter(s['sample_rates_hz']))} Hz mono audio. The normalized signal range is {inventory.signal_min.min():.5f} to {inventory.signal_max.max():.5f}. There are {q['missing_label']} missing labels and {q['duplicate_groups']} exact PCM-copy groups.",size=9.7)
    r.p('Copies occur within the same speaker, class, and split. Before modeling, retain the first path alphabetically in each group and update the effective class counts. Energy and saturation flags remain available for reviewing recording quality.',size=9.5)
    r.latin('Evidence: quality_review_queue.csv / duplicate_records.csv / audio_inventory.csv',size=7.8)
    r.section('Why these screening thresholds?')
    r.p('**Energy.** Clip RMS below -45 dBFS (0.56% of full scale) prioritizes very quiet recordings for listening. The lower -50 dBFS frame threshold (0.32%) summarizes quiet 20 ms intervals within clips. Both are amplitude screens, not speech or SNR labels.',size=9.3,leading=14,gap=6)
    r.p('**Length.** The 0.25 s rule marks clips occupying less than a quarter of the nominal one-second window for review. A short command can still be valid.',size=9.3,leading=14,gap=6)
    r.p('**DC offset.** An absolute mean above 0.05 flags a baseline shift exceeding 5% of full scale, which can reduce amplitude headroom. Review the waveform before deciding on correction.',size=9.3,leading=14,gap=6)

    r.new('Class Balance Across Partitions')
    r.figure('class_split_counts',max_h=260)
    r.section('Reading the class counts')
    r.analysis(
        f"Every command has 1,000 clips overall. Training counts range from {counts.train.min()} to {counts.train.max()}; validation and test also show small class differences after grouping by speaker.",
        'The corpus is balanced by command, while the grouped split preserves related recordings together. The small partition differences are compatible with that grouping and do not justify resampling by themselves.',
        'Keep the speaker-based split. Start without class resampling, use Macro-F1 alongside per-class recall, and recompute effective counts after exact-copy removal.')
    r.p('The colored segments are counts, not recognition scores. This figure describes the data available for evaluation; model performance will be measured in later phases.',size=9.3,leading=15)

    r.new('Training Distributions: Length and Amplitude')
    r.figure('train_distributions',max_h=210)
    r.table(['Train variable','Mean +/- SD','Median','95th pct.','Maximum'],[
        [name,f'{stats.loc[k,"mean"]:.3f} +/- {stats.loc[k,"std"]:.3f}',f'{stats.loc[k,"50%"]:.3f}',f'{stats.loc[k,"95%"]:.3f}',f'{stats.loc[k,"max"]:.3f}']
        for k,name in [('duration_s','Duration (s)'),('rms_dbfs','Energy (dBFS)'),('peak_abs','Peak amplitude')]
    ],widths=[CW*.25,CW*.30,CW*.15,CW*.15,CW*.15],size=8.5)
    r.analysis(
        f"Training file lengths range from {stats.loc['duration_s','min']:.3f} to {stats.loc['duration_s','max']:.3f} s, with a median of one second. Energy has a standard deviation of {stats.loc['rms_dbfs','std']:.2f} dB, and peak amplitudes extend to full scale.",
        'The one-second concentration mainly describes the recording format; it does not measure how long a word is spoken. Wide amplitude variation makes recording level a possible source of variation in audio features.',
        'Pad shorter clips to 16,000 samples and retain their original lengths. Compare feature scaling and gain augmentation using training data, then check validation results separately for low-energy clips.')

    r.new('Command, Duration, and Energy Relationships')
    r.figure('train_class_relationships',max_h=190)
    med = train.groupby('label')[['duration_s','rms_dbfs','low_energy_frame_fraction']].median()
    low,high = med.rms_dbfs.idxmin(),med.rms_dbfs.idxmax()
    short_rates = (train.n_frames < train.sample_rate_hz).groupby(train.label).mean() * 100
    short_low,short_high = short_rates.idxmin(),short_rates.idxmax()
    r.analysis(
        f"Short-clip rates range from {short_rates[short_low]:.1f}% ({short_low}) to {short_rates[short_high]:.1f}% ({short_high}), using each command's training count as denominator. Median energy ranges from {med.loc[low,'rms_dbfs']:.2f} to {med.loc[high,'rms_dbfs']:.2f} dBFS; energy boxes overlap.",
        'Short-clip rates reveal length differences hidden by the one-second median. Energy and quiet-frame distributions show substantial within-command variation.',
        'Pad short clips and compare these summaries with spectral features. Full counts are in train_short_clip_rates.csv; boxplot outliers remain in the inventory.',size=9.5,leading=14)
    r.figure('train_multivariate',max_h=210)
    corr = pd.read_csv(OUT/'train_spearman.csv',index_col=0)
    r.analysis(
        f"Spearman correlations are {corr.loc['rms_dbfs','peak_abs']:.2f} for energy/peak, {corr.loc['rms_dbfs','low_energy_frame_fraction']:.2f} for energy/quiet-frame fraction, and {corr.loc['duration_s','rms_dbfs']:.2f} for duration/energy.",
        'Energy and peak carry overlapping amplitude information. The quiet-frame association partly reflects its amplitude threshold; file length has little rank association with energy.',
        'Fit scaling on train and compare feature sets with and without correlated amplitude summaries before PCA or clustering.',size=9.5,leading=14)
    r.latin('Relationship plots use the training set.',size=8.2)

    r.new('Speaker Audit and Command Coverage')
    r.figure('speaker_coverage',max_h=220)
    quant = s['speaker_recording_quantiles']
    r.table(['Distinct commands per ID','Number of IDs'],[[str(k),str(v)] for k,v in s['speaker_label_counts'].items()],widths=[CW*.58,CW*.42],row_h=24)
    coverage = {int(k):v for k,v in s['speaker_label_counts'].items()}
    r.analysis(
        f"Clips per ID have median {quant['0.5']:.0f}, 95th percentile {quant['0.95']:.0f}, and maximum {quant['1.0']:.0f}. Of {s['n_speakers']:,} IDs, {sum(v for k,v in coverage.items() if k <= 3)/s['n_speakers']:.1%} cover at most three commands; only {coverage[8]} cover all eight.",
        'Balanced commands do not imply balanced speaker contributions. Frequent IDs receive more weight in clip-level scores, and incomplete command coverage can mix speaker-related variation with command differences.',
        'Preserve speaker grouping, inspect per-class results, and use speaker-level bootstrap sampling for uncertainty. Keep speaker IDs out of prediction features.')
    r.p('speakers.csv and speaker_class_counts.csv retain the complete contribution and coverage tables. PCM hashes supply the separate check for exact audio copies.',size=9.2,leading=14)

    r.new('Partition Design and Information Leakage')
    r.p("TensorFlow's which_set rule hashes the speaker prefix with SHA-1. The first 10% of hash space is assigned to validation, the next 10% to test, and the rest to training. A speaker keeps the same assignment when more files are added.")
    r.table(['Partition','Valid clips','Speaker/session IDs'],[[sp,str(s['splits'][sp]['recordings']),str(s['splits'][sp]['speakers'])] for sp in ['train','validation','test']],widths=[CW*.34,CW*.33,CW*.33],row_h=27)
    r.table(['Command','Train','Validation','Test'],[[l,str(counts.loc[l,'train']),str(counts.loc[l,'validation']),str(counts.loc[l,'test'])] for l in cfg['labels']],widths=[CW*.3,CW*.25,CW*.25,CW*.2],row_h=24)
    r.section('Evidence from a shuffled clip-level control')
    r.p(f"A shuffled clip-level split places familiar speakers in the evaluation sets: {pct(s['naive_row_split']['validation_seen_speaker_fraction'])} of validation clips and {pct(s['naive_row_split']['test_seen_speaker_fraction'])} of test clips share an ID with training. The speaker-based split avoids this overlap.")
    r.section('Training and evaluation')
    r.p('All derived copies, padding, and augmentations stay in the source partition. Fit scalers, PCA, and feature-selection methods on train; tune on validation; reserve test for final evaluation. Paths, speaker IDs, class names, and test-derived statistics must not enter prediction features.',size=9.8)
    r.latin('Checks: 0 shared IDs; 0 cross-partition exact PCM-copy groups.',size=8)

    r.new('Noise, Silence, and Low-Energy Clips')
    r.table(['Component','Observed count / availability'],[
        ['Independent _background_noise_ files',str(s['background_noise_files'])],
        ['Separate silence / unknown classes','Absent'],
        ['Clips with every sample exactly zero',str(q['exact_silence'])],
        ['Clips below -45 dBFS RMS',str(q['low_energy'])],
        ['Provided validation/testing lists','Absent']
    ],widths=[CW*.65,CW*.35],row_h=29)
    r.section('What does the measurement establish?')
    r.p('RMS is calculated after dividing PCM values by 32768. The quiet-frame measure counts nonoverlapping 20 ms frames below -50 dBFS. It describes the amplitude pattern within a clip.')
    r.section('Decision for the classification task')
    r.p('Low-energy clips keep their command labels. The review queue lists them alongside the other quality flags. Preprocessing decisions will be made in the next phase.')
    r.section('Limit and later evaluation design')
    r.p('Noise experiments will use documented sources and fixed SNR values. Training noise sources will be separate from evaluation sources. Independent noise and unrelated speech are also needed to measure false activations.')
    r.p('The following pages show three fixed training examples per command. They illustrate timing and spectral variation rather than estimate class-wide patterns. A full listening review has not been completed.',size=9.8)

    comments = {
        'down': (
            'The first and third recordings have stronger activity early in the clip, while the second has a weak, later event. Its waveform is almost flat at the shared scale, yet the spectrum retains visible structure.',
            'A low waveform amplitude does not by itself establish an empty recording. The same label also occurs at different temporal positions, so fixed onset assumptions could discard useful content.',
            'Review the low-energy example by listening, keep its label during the audit, and compare spectral features with training-only time shifts.'),
        'go': (
            'The first example has a much larger waveform than the other two. Low-frequency bands are visible in all three; the third also has diffuse spectral energy extending beyond its main burst.',
            'Within-command variation includes both recording level and the surrounding signal. A single RMS value combines these components and cannot describe their time-frequency arrangement.',
            'Compare MFCC or log-mel features with amplitude summaries. Evaluate gain augmentation on train and inspect validation errors by energy level.'),
        'left': (
            'The first two recordings show weak activity later in the clip. The third has stronger early activity, a later burst, and a sharp transient close to the end.',
            'The largest peak is not necessarily the part that best represents the command. Peak-based alignment or normalization could give an isolated transient too much influence.',
            'Listen to the transient-containing example before trimming. Compare robust feature scaling and full-clip spectral features before adopting peak-based preprocessing.'),
        'no': (
            'The main activity appears in the middle-to-late part of these examples. Their spectra show horizontal bands whose frequency distribution changes over time; the third also has a short earlier transient.',
            'Clips with the same word can differ in timing and spectral shape. A whole-clip energy statistic loses the ordering and frequency detail visible in these panels.',
            'Retain the full clip with padding and compare temporal spectral features against summary features. Review isolated transients before defining a trimming rule.'),
        'right': (
            'The second example has a small late burst and a long low-amplitude interval. The first and third show stronger activity, with different burst widths and frequency-band patterns.',
            'The active interval varies even within one command. Absolute position and the amount of quiet audio are therefore fragile cues for a model intended for unseen speakers.',
            'Compare train-only time shifts and temporal pooling while preserving the full event. Use per-class validation recall to assess whether right benefits from these choices.'),
        'stop': (
            'The first example concentrates its strongest activity near the end. The second ends at about 0.65 s; the white region to its right is outside the recorded signal. The third has visible energy throughout the clip.',
            'Shorter file duration and low signal energy are different conditions. The unrecorded tail of the second example should not be treated as measured silence or background noise.',
            'Pad short clips while retaining original duration metadata. Review background activity by listening and inspect stop-command recall and errors in later modeling.'),
        'up': (
            'The first example has a strong early burst, the second concentrates activity near the one-second boundary, and the third has a smaller event around the middle of the clip.',
            'One-second files do not imply equal speech duration or a common onset. A fixed central crop could remove much of the late event in the second example.',
            'Keep the complete one-second input and avoid a central crop. If using time shifts, limit them to training and check that active content remains inside the window.'),
        'yes': (
            'The examples differ in peak timing and the number of visible bursts. Their spectra contain lower-frequency bands and later, broader high-frequency regions. The third clip contains two endpoint PCM samples.',
            'Time-frequency structure supplies information beyond overall amplitude. Reaching a PCM endpoint is a quality-review flag; its effect on the recording requires closer inspection.',
            'Review the endpoint samples and compare spectral representations. Later, inspect the confusion matrix to identify which commands are confused with yes.')
    }
    examples = pd.read_csv(OUT/'example_selection.csv')
    for label in cfg['labels']:
        r.new(f'Waveforms and Spectra: {label.upper()}', 'Three fixed training recordings; illustrative examples')
        r.figure('examples_'+label,width=CW,max_h=350)
        subset = examples[examples.label == label]
        r.p(f"Selected-example RMS range: {subset.rms_dbfs.min():.2f} to {subset.rms_dbfs.max():.2f} dBFS.",size=8.7,leading=13,gap=10)
        r.analysis(*comments[label])
        r.p('STFT: Hann window 400, hop 160, FFT 512. Shared -100 to 0 dB color scale; waveforms share the -1 to 1 amplitude scale.',size=8.3,leading=12,gap=3)

    r.new('Baseline and Next Steps')
    b = s['baseline']
    r.section('Simple Phase 1 control')
    r.p(f"The constant prediction is the most frequent training class, {b['prediction']}, with {b['training_count']} training clips. The class is not selected using validation or test. Lexicographic order breaks ties.")
    r.table(['Metric','Validation value'],[
        ['Macro-F1',f"{b['validation_macro_f1']:.4f}"],
        ['Balanced accuracy',f"{b['validation_balanced_accuracy']:.4f}"],
        ['Accuracy',f"{b['validation_accuracy']:.4f}"],
        ['Test model score','Not computed']
    ],widths=[CW*.55,CW*.45],row_h=29)
    r.analysis(
        f"Always predicting {b['prediction']} gives {b['validation_accuracy']:.1%} validation accuracy and {b['validation_macro_f1']:.4f} Macro-F1, with balanced accuracy of {b['validation_balanced_accuracy']:.3f}.",
        'Accuracy equals the validation prevalence of stop, which is below the 12.5% expected accuracy of uniform guessing. Only one command receives any recall, explaining the low Macro-F1.',
        'Require future models to improve across commands using Macro-F1 and per-class recall, then examine the confusion matrix rather than relying on overall accuracy alone.')
    r.section('Plan for the next phase')
    r.p('Phase 2: standardize length and rate; build MFCC or log-mel features; fit scaling on train; apply PCA and k-means; assess silhouette and stability across seeds. Examine cluster relationships with command, speaker/session, and energy without fitting the unsupervised method to command labels.')
    r.p('Modeling: compare a classical MFCC-summary/logistic-regression baseline with an audio model. Primary metric: Macro-F1. Also report balanced accuracy, accuracy, per-class precision/recall, and a confusion matrix. Analyze stop-command errors and opposing command pairs separately.')
    r.p('Further experiments will estimate uncertainty with speaker-level bootstrap sampling, test noise sensitivity at specified SNR values, and compare a feature or augmentation ablation.',size=9.8)

    r.new('Reproduction, Limitations, and References')
    r.section('Reproduction pipeline')
    for line in ['python -m pip install -r requirements.txt','python src/download_data.py',
                 'python src/audit.py','python src/build_report.py']:
        r.latin(line,size=9,gap=19)
    r.p('The pipeline writes reports/phase1.pdf. The configuration records the source, thresholds, seed, and spectral settings. The CSV files contain the measurements, split assignments, and summary statistics used in the report.',size=9.8)
    r.section('Limitations')
    r.p('The mini dataset contains prompted English words and provides limited recording metadata. Demographics, devices, and physical speaker identities were not independently verified. Exact checksums cover identical copies rather than all near-duplicates. Energy flags are amplitude measures, not SNR estimates. Continuous-listening performance will require additional data.',size=9.8)
    r.p('The screening cut-offs are exploratory choices rather than calibrated detection thresholds. A later listening review and sensitivity analysis should guide any exclusion or correction.',size=9.3,leading=14)
    r.p('All reported measurements were calculated from the mini dataset. Report updated: 4 November 2026.',size=9,leading=14)
    r.section('References')
    references = [
        ('1. Pete Warden (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209.',
         'https://arxiv.org/abs/1804.03209',
         'Dataset collection, verification, and evaluation background.'),
        ('2. TensorFlow. Simple audio recognition: Recognizing keywords.',
         'https://www.tensorflow.org/tutorials/audio/simple_audio',
         'Official mini archive, eight-command layout, and audio format.'),
        ('3. TensorFlow Datasets. speech_commands dataset catalog.',
         'https://www.tensorflow.org/datasets/catalog/speech_commands',
         'Parent dataset configuration; distinct from the mini archive analyzed here.'),
    ]
    for citation,url,purpose in references:
        r.p(citation,size=9.1,leading=14,gap=4)
        r.link(url,url,size=8.3)
        r.p(purpose+' Accessed 4 November 2026.',size=8.5,color=MUTED,leading=13,gap=12)
    r.save()
    build_path.replace(final_path)
    manifest = {'language':'en','pages':len(r.pages),'page_titles':r.pages,
                'pdf_file':final_path.relative_to(ROOT).as_posix(),
                'layout':{'font':'Helvetica / Helvetica-Bold',
                          'heading_color':'#16364d','accent_color':'#167d8d',
                          'body_color':'#263849','table_tint':'#eff4f7',
                          'numbered_figures':r.figure_number},
                'content_bottoms_pt':r.page_bottoms,
                'pdf_sha256':hashlib.sha256(final_path.read_bytes()).hexdigest(),
                'dataset_sha256':s['dataset_sha256'],'config_sha256':s['config_sha256']}
    (BUILD/'report.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Wrote {final_path.relative_to(ROOT).as_posix()} ({len(r.pages)} English pages)')


if __name__ == '__main__':
    main()
