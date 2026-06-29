# معماری علمی پیشنهادی پروژه

## 1. ضربان قلب

مسیر اصلی:

```text
video -> multi-ROI RGB traces -> classical rPPG priors -> waveform model -> rPPG waveform -> HR
```

### چرا waveform-first؟

اگر مدل مستقیم HR بدهد، ممکن است shortcut یاد بگیرد. مثلا به جای یادگیری سیگنال فیزیولوژیک، از bias دیتاست یا نور یا حرکت عدد HR را حدس بزند. اما اگر waveform را پیش‌بینی کند، می‌توانیم هم فرکانس را بسنجیم، هم correlation waveform، هم SNR و هم Bland-Altman روی HR را.

### مدل‌های موجود در کد

- `HR1DCNN`: فقط baseline مستقیم HR.
- `WaveformNet`: مدل اصلی فعلی برای یک ROI یا RGB+prior.
- `MultiROIWaveformNet`: نسخه آماده برای ROI attention. ورودی آن `(B, R, C, L)` است.

### ورودی پیشنهادی فعلی

برای شروع:

```text
C = 8 = RGB(3) + POS_WIN + CHROM_WIN + LGI + OMIT + PBV
```

بعدا برای multi-ROI:

```text
R = forehead, left_cheek, right_cheek, face
C = RGB + priors per ROI
```

## 2. فشارخون

مسیر علمی پیشنهادی:

```text
face rPPG + hand rPPG + PTT/phase + PWA + HR + metadata + cuff calibration -> SBP/DBP + uncertainty
```

مدل face-only BP فقط baseline است و نباید claim اصلی پروژه باشد.

## 3. معیارهای ارزیابی

برای HR:

- MAE/RMSE bpm
- Pearson correlation
- Bland-Altman
- waveform correlation
- SNR
- cross-dataset test

برای BP:

- ME ± SD
- MAE/RMSE
- Bland-Altman
- subject-wise یا cohort-wise split
- calibration-aware evaluation
- sanity check نزدیک به AAMI: `abs(ME)<=5` و `SD<=8`

## 4. قانون‌های پروژه

1. window split برای نتیجه نهایی ممنوع.
2. هر مدل BP باید baseline mean/person-calibration را شکست دهد.
3. هر claim BP باید uncertainty یا quality flag داشته باشد.
4. اول robust rPPG، بعد BP.
