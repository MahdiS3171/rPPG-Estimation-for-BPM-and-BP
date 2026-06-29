# TODO مرحله‌ای

## مرحله 1: sanity check کد

- نصب requirements
- اجرای `extract_rppg_from_video.py` روی یک ویدیو کوتاه
- بررسی اینکه MediaPipe درست face را پیدا می‌کند

## مرحله 2: baseline کلاسیک

- گرفتن UBFC-rPPG
- اجرای POS_WIN، CHROM_WIN، LGI، OMIT، PBV
- ساخت جدول MAE/RMSE و Bland-Altman

## مرحله 3: WaveformNet

- آموزش با RGB فقط
- آموزش با RGB + priors
- مقایسه subject-wise
- بررسی waveform correlation و HR MAE

## مرحله 4: Multi-ROI

- استخراج ROI جداگانه
- فعال‌سازی `MultiROIWaveformNet`
- تحلیل attention روی ROIها

## مرحله 5: فشارخون pilot

- انتخاب گوشی 120/240fps
- تست timestamp با LED/metronome نوری
- طراحی recording protocol
- ثبت pilot با face+hand+contact PPG+cuff
- تست PTT/phase قبل از آموزش مدل BP
