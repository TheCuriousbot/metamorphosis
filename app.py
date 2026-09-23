import io, json, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, streamlit as st
from core import analyze_field, aggregate_fields, overlay
from report import build_pdf

st.set_page_config(page_title='Cast Iron ISO 945 Analyzer V6', layout='wide')
st.title('Cast Iron Graphite / Nodularity Analyzer — V6')
st.caption('ISO 945-4:2019-aligned prototype • fast multi-scale analysis • adaptive segmentation • adaptive/controlled particle separation • PDF reporting')

if 'last_results' not in st.session_state: st.session_state.last_results=None

with st.sidebar:
    st.header('Analysis settings')
    engine = st.selectbox('Segmentation engine', ['auto-stable','iso-enhanced','iso-reference','blackhat'],
                          format_func=lambda x:{'auto-stable':'AUTO — stable multi-method','iso-enhanced':'Enhanced Otsu + local rescue','iso-reference':'Public Otsu reference baseline','blackhat':'Local-darkness rescue'}[x])
    scale = st.number_input('Pixel calibration (µm/pixel)', min_value=0.0, value=0.0, step=0.001, format='%.4f')
    field_width_mm = st.number_input('OR field width (mm)', min_value=0.0, value=0.0, step=0.01, format='%.3f')
    min_size = st.selectbox('Minimum maximum-Feret diameter', [10.0,5.0], index=0, format_func=lambda x:f'{x:.0f} µm')
    magnification = st.number_input('Magnification', min_value=1, value=100, step=1)
    separate = st.checkbox('Separate joined particles', True)
    sep_mode = st.selectbox('Separation', ['auto','manual'], format_func=lambda x:'AUTO — test 0.50 / 0.60 / 0.70' if x=='auto' else 'Manual')
    peak_ratio = st.slider('Separation peak ratio',0.25,0.90,0.70,0.01,disabled=(sep_mode=='auto'))
    peak_dist = st.slider('Minimum peak distance (px)',3,30,8,1)
    max_dim = st.select_slider('Maximum analysis dimension', options=[1200,1600,1800,2200,2600], value=1800,
                                help='Large microscope images are automatically downscaled for speed; calibration is adjusted so measurements remain in µm.')
    st.divider(); st.info('ISO 945-4:2019 specifies representative evaluation using at least five fields and about 500 graphite particles. A single field is useful for screening but is not a complete ISO evaluation.')

def read_img(upload): return cv2.imdecode(np.frombuffer(upload.getvalue(),np.uint8),cv2.IMREAD_COLOR)
def effective_scale(img):
    if scale>0:return float(scale)
    if field_width_mm>0:return field_width_mm*1000.0/img.shape[1]
    return None

def analyze_one(img):
    if sep_mode=='auto':
        candidates=[]
        for pr in (0.50,0.60,0.70):
            f=analyze_field(img,engine,effective_scale(img),min_size,separate,pr,peak_dist,max_dim)
            # Internal quality: penalize excessive tiny particles and reward stable coverage/roundness distribution.
            rs=f['rows']; tiny=sum(r['feret_um']<min_size*1.15 for r in rs)/max(1,len(rs))
            elongated=sum(r['aspect_ratio']>4 for r in rs)/max(1,len(rs))
            q=100-80*tiny-25*elongated
            f['_auto_quality']=q; f['_auto_peak_ratio']=pr; candidates.append(f)
        # Prefer highest internal quality; when close, prefer 0.70 (user's observed successful setting).
        candidates.sort(key=lambda x:(x['_auto_quality'],x['_auto_peak_ratio']),reverse=True)
        return candidates[0]
    return analyze_field(img,engine,effective_scale(img),min_size,separate,peak_ratio,peak_dist,max_dim)

st.header('1. Analyze microscope fields')
ups=st.file_uploader('Upload one or more microscope fields',type=['png','jpg','jpeg','tif','tiff'],accept_multiple_files=True)
if ups:
    sample_id=st.text_input('Sample / casting ID','Sample-001')
    c1,c2=st.columns(2); location=c1.text_input('Sampling location',''); material=c2.text_input('Material designation','Spheroidal graphite cast iron')
    c3,c4=st.columns(2); analyst=c3.text_input('Analyst',''); laboratory=c4.text_input('Laboratory','')
    fields=[]; start=time.time(); prog=st.progress(0)
    for i,u in enumerate(ups):
        img=read_img(u); f=analyze_one(img); f['filename']=u.name; f['display_image']=img; fields.append(f); prog.progress((i+1)/len(ups))
    prog.empty(); agg=aggregate_fields(fields); st.session_state.last_results=fields
    st.subheader('Aggregate result')
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric('Nodularity',f"{agg['nodularity']:.1f}%"); m2.metric('Graphite particles',agg['particle_count']); m3.metric('Particles/mm²',f"{agg['particle_count_per_mm2']:.1f}" if agg['particle_count_per_mm2'] else 'N/A'); m4.metric('Graphite size',str(agg['dominant_size']) if agg['dominant_size'] else 'N/A'); m5.metric('Analysis time',f'{time.time()-start:.1f}s')
    if not agg['full_sampling_requirement_ok']: st.warning(f"Sampling status: {agg['fields']} field(s), {agg['particle_count']} particles. ISO 945-4 recommends at least five fields and about 500 particles.")
    else: st.success('Prototype sampling minimum reached.')
    for i,f in enumerate(fields,1):
        with st.expander(f"Field {i}: {f['filename']}",expanded=(i==1)):
            a,b,c=st.columns(3); a.image(cv2.cvtColor(f['display_image'],cv2.COLOR_BGR2RGB),caption='Original',use_container_width=True); b.image(f['mask'],caption='Graphite mask used for analysis',use_container_width=True); c.image(cv2.cvtColor(overlay(f['display_image'],f),cv2.COLOR_BGR2RGB),caption='Particle classification overlay',use_container_width=True)
            st.write({'nodularity_%':round(f['nodularity'],3),'particles':f['particle_count'],'largest_Feret_um':f['feret_max_um'],'mean_roundness':round(f['roundness_mean'],4),'splits':f['split_meta']['splits'],'segmentation':f['segmentation'].get('method'),'speed':f.get('speed_note'),'auto_peak_ratio':f.get('_auto_peak_ratio','manual')})
            st.download_button('Download particle CSV',pd.DataFrame(f['rows']).to_csv(index=False).encode(),f"{Path(f['filename']).stem}_particles.csv",'text/csv',key=f'csv{i}')
    if st.button('Generate ISO 945-4 prototype report PDF',type='primary'):
        meta={'sample_id':sample_id,'location':location,'material':material,'analyst':analyst,'laboratory':laboratory,'date':pd.Timestamp.now().strftime('%Y-%m-%d'),'engine':engine,'magnification':magnification,'min_size':min_size,'scale':effective_scale(fields[0]['display_image']),'peak_ratio':peak_ratio if sep_mode=='manual' else 'AUTO','max_dim':max_dim}
        pdf=build_pdf(fields,agg,meta); st.download_button('Download PDF report',pdf,f"{sample_id}_ISO9454_V6_report.pdf",'application/pdf',type='primary'); st.success('PDF report ready.')

st.divider(); st.header('2. Known-reference calibration')
st.write('Use this only when you have a trusted reference nodularity for an image. It searches segmentation + separation settings against that known value; it does not manufacture a ground truth.')
cal=st.file_uploader('Calibration image',type=['png','jpg','jpeg','tif','tiff'],key='calimg')
if cal:
    target=st.number_input('Known nodularity (%)',0.0,100.0,60.0,0.5)
    tol=st.number_input('Acceptable target tolerance (percentage points)',0.5,20.0,5.0,0.5)
    if st.button('Run calibration search'):
        img=read_img(cal); rows=[]; best=None
        with st.spinner('Testing calibration candidates...'):
            for eng in ['iso-reference','iso-enhanced','blackhat']:
                for pr in [0.50,0.55,0.60,0.65,0.70,0.75]:
                    f=analyze_field(img,eng,effective_scale(img),min_size,True,pr,peak_dist,max_dim)
                    err=abs(f['nodularity']-target); rows.append({'engine':eng,'peak_ratio':pr,'nodularity_%':f['nodularity'],'absolute_error_pp':err,'particles':f['particle_count'],'splits':f['split_meta']['splits']})
        df=pd.DataFrame(rows).sort_values(['absolute_error_pp','particles']).reset_index(drop=True); st.dataframe(df,use_container_width=True)
        best=df.iloc[0]; st.success(f"Best tested setting: {best.engine}, peak ratio {best.peak_ratio:.2f} → {best['nodularity_%']:.2f}% (error {best.absolute_error_pp:.2f} percentage points).")
        st.download_button('Download calibration results CSV',df.to_csv(index=False).encode(),'V6_calibration_results.csv','text/csv')

st.divider(); st.header('3. What data improves V6')
st.markdown('''Provide **5–15 real microscope fields** if possible. For each image, the most useful ground truth is: **(1)** trusted nodularity %, **(2)** microscope magnification and pixel/µm calibration or field width, **(3)** nodule/graphite particle count if independently measured, and **(4)** a note if the image is a difficult case (touching nodules, flakes, uneven illumination, low contrast, etc.). A few manually corrected particle masks/labels are even more valuable than many synthetic images. The goal is to tune against real failure modes, not overfit synthetic data.''')
