import io, cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from core import overlay

def _imgbio(im):
    ok,enc=cv2.imencode('.jpg',im,[int(cv2.IMWRITE_JPEG_QUALITY),88]); b=io.BytesIO(enc.tobytes()); b.seek(0); return b

def build_pdf(fields,agg,meta):
    buf=io.BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=12*mm,leftMargin=12*mm,topMargin=12*mm,bottomMargin=12*mm)
    s=getSampleStyleSheet(); story=[Paragraph('CAST IRON GRAPHITE / NODULARITY TEST REPORT',s['Title']),Paragraph('ISO 945-4:2019-aligned image-analysis prototype — V6',s['Heading2']),Spacer(1,4*mm)]
    status='Sampling minimum reached' if agg.get('full_sampling_requirement_ok') else 'LIMITED FIELD SET — minimum sampling not reached'
    rows=[['Sample / casting ID',meta['sample_id']],['Sampling location',meta['location']],['Material designation',meta['material']],['Analyst',meta['analyst']],['Laboratory',meta['laboratory']],['Date',meta['date']],['Software','Cast Iron ISO 945 Analyzer V6'],['Segmentation engine',meta['engine']],['Magnification',f"×{meta['magnification']}"],['Minimum maximum-Feret size',f"{meta['min_size']:.0f} µm"],['Pixel calibration',f"{meta['scale']:.4f} µm/pixel" if meta['scale'] else 'Not supplied'],['Separation peak ratio',str(meta['peak_ratio'])],['Analysis max dimension',str(meta['max_dim'])+' px'],['Fields evaluated',str(agg['fields'])],['Graphite particles evaluated',str(agg['particle_count'])],['Sampling status',status]]
    t=Table(rows,colWidths=[65*mm,115*mm]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(0,-1),colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP')])); story += [t,Spacer(1,4*mm)]
    summary=[['Parameter','Result'],['Nodularity',f"{agg['nodularity']:.1f} %"],['Graphite particle count',str(agg['particle_count'])],['Particles/mm²',f"{agg['particle_count_per_mm2']:.1f}" if agg['particle_count_per_mm2'] is not None else 'N/A'],['Graphite size class',str(agg['dominant_size']) if agg['dominant_size'] else 'N/A'],['Largest maximum-Feret',f"{agg['largest_feret_um']:.1f} µm" if agg['largest_feret_um'] else 'N/A'],['Mean maximum-Feret',f"{agg['feret_mean_um']:.1f} µm" if agg['feret_mean_um'] else 'N/A'],['Feret SD',f"{agg['feret_std_um']:.1f} µm"],['Mean roundness',f"{agg['roundness_mean']:.3f}"]]
    st=Table(summary,colWidths=[80*mm,100*mm]); st.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.darkgrey),('TEXTCOLOR',(0,0),(-1,0),colors.white)])); story += [st,Spacer(1,4*mm),Paragraph('Method / conformity notes',s['Heading2']),Paragraph('The prototype uses an open-source/public Otsu-contour baseline as one segmentation option, removes field-border particles, applies a maximum-Feret size filter, separates selected contiguous particles using marker-controlled watershed, and calculates roundness as 4A/(π·maximum-Feret²). Graphite with roundness ≥ 0.60 is classified as spheroidal for the ISO-aligned area calculation. ISO 945-4:2019 specifies representative examination using at least five fields and about 500 graphite particles; this report flags whether that sampling condition was reached.',s['BodyText']),Spacer(1,2*mm),Paragraph('This document is an ISO 945-4-aligned prototype report, not an accredited/certified conformity statement. Production QC use requires validation against qualified reference specimens, controlled sample preparation, calibrated imaging and documented laboratory procedures.',s['BodyText']),Spacer(1,4*mm)]
    sz=[['ISO size class','Count']]+[[str(k),str(agg['size_counts'].get(str(k),0))] for k in range(3,9)]; z=Table(sz,colWidths=[80*mm,100*mm]); z.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)])); story += [Paragraph('Graphite size distribution',s['Heading2']),z,PageBreak()]
    for i,f in enumerate(fields,1):
        display=f.get('display_image',f['image']); story.append(Paragraph(f"Field {i}: {f.get('filename','')}",s['Heading2']))
        for im,cap in [(display,'Original micrograph'),(f['mask'],'Graphite segmentation'),(overlay(display,f),'Particle overlay')]:
            story.append(Paragraph(cap,s['BodyText'])); b=_imgbio(im); story.append(RLImage(b,width=175*mm,height=175*mm*im.shape[0]/im.shape[1])); story.append(Spacer(1,2*mm))
        story.append(Paragraph(f"Field nodularity {f['nodularity']:.2f}% | particles {f['particle_count']} | largest Feret {f['feret_max_um']:.1f} µm | splits {f['split_meta']['splits']} | segmentation {f['segmentation'].get('method','')}",s['BodyText']))
        if i<len(fields):story.append(PageBreak())
    doc.build(story); return buf.getvalue()
