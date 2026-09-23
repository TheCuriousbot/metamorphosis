import cv2
import numpy as np
import math

ISO_SIZE_RANGES_UM = {3:(250.0,500.0),4:(120.0,250.0),5:(60.0,120.0),6:(30.0,60.0),7:(15.0,30.0),8:(0.0,15.0)}


def max_feret_from_contour(contour):
    hull = cv2.convexHull(contour)[:,0,:].astype(np.float64)
    n=len(hull)
    if n<2: return 0.0
    # Downsample very dense convex hulls for speed while preserving the outer envelope.
    if n>700:
        idx=np.linspace(0,n-1,700).astype(int); hull=hull[idx]; n=len(hull)
    # Exact-ish rotating-calipers diameter for convex polygon.
    j=1; best2=0.0
    for i in range(n):
        ni=(i+1)%n; edge=hull[ni]-hull[i]
        while True:
            nj=(j+1)%n
            c1=abs(edge[0]*(hull[j][1]-hull[i][1])-edge[1]*(hull[j][0]-hull[i][0]))
            c2=abs(edge[0]*(hull[nj][1]-hull[i][1])-edge[1]*(hull[nj][0]-hull[i][0]))
            if c2>c1: j=nj
            else: break
        for q in (j,ni):
            d=hull[i]-hull[q]; best2=max(best2,float(np.dot(d,d)))
    return math.sqrt(best2)


def contour_measurements(mask, scale_um_per_px=1.0):
    m=(mask>0).astype(np.uint8)*255
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return None
    c=max(contours,key=cv2.contourArea); area_px=float(cv2.contourArea(c)); feret_px=max_feret_from_contour(c)
    roundness=4*area_px/(math.pi*feret_px**2) if feret_px>0 else 0.0
    per=float(cv2.arcLength(c,True)); circularity=4*math.pi*area_px/(per*per) if per>0 else 0.0
    hull=cv2.convexHull(c); ha=float(cv2.contourArea(hull)); solidity=area_px/ha if ha>0 else 0.0
    x,y,w,h=cv2.boundingRect(c)
    return {'area_px':area_px,'area_um2':area_px*scale_um_per_px**2,'feret_px':feret_px,'feret_um':feret_px*scale_um_per_px,
            'roundness':float(roundness),'circularity':float(circularity),'solidity':float(solidity),
            'aspect_ratio':max(w,h)/max(1.0,min(w,h)),'bbox_x':int(x),'bbox_y':int(y),'bbox_w':int(w),'bbox_h':int(h)}


def size_class(feret_um):
    if feret_um>=250:return 3
    if feret_um>=120:return 4
    if feret_um>=60:return 5
    if feret_um>=30:return 6
    if feret_um>=15:return 7
    return 8


def _remove_border(mask):
    m=(mask>0).astype(np.uint8); n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    border=set(np.unique(lab[0]).tolist()+np.unique(lab[-1]).tolist()+np.unique(lab[:,0]).tolist()+np.unique(lab[:,-1]).tolist())
    out=m.copy()
    for b in border:
        if b: out[lab==b]=0
    return out*255


def _clean(mask,min_component_px=5):
    mask=_remove_border(mask)
    if min_component_px>1:
        n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]<min_component_px: mask[lab==i]=0
    return mask


def segment_reference_otsu(img,min_component_px=5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    t,b=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return _clean(b,min_component_px),{'method':'Reference Otsu','threshold':int(t)}


def segment_enhanced_otsu(img,min_component_px=5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bg=cv2.GaussianBlur(gray,(0,0),25)
    corr=cv2.normalize(bg.astype(np.float32)-gray.astype(np.float32),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    _,local=cv2.threshold(corr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t,g=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    local=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),1)
    return _clean(cv2.bitwise_or(g,local),min_component_px),{'method':'Enhanced Otsu + local rescue','threshold':int(t)}


def segment_blackhat(img,min_component_px=5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Multi-scale local darkness; deliberately no closing on the final mask.
    k=max(15,int(round(min(gray.shape[:2])*0.015))|1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    bh=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
    _,m=cv2.threshold(bh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),1)
    return _clean(m,min_component_px),{'method':'Blackhat local-darkness','kernel':k}


def _mask_stats(mask):
    frac=float((mask>0).mean())
    n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    areas=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.array([])
    return frac,int(len(areas)),float(np.median(areas)) if len(areas) else 0.0


def _split_by_distance(comp,peak_ratio=0.70,min_peak_distance_px=8):
    d=cv2.distanceTransform(comp,cv2.DIST_L2,5); mx=float(d.max())
    if mx<2:return [comp],False
    dil=cv2.dilate(d,np.ones((3,3),np.uint8)); peaks=((d>=dil-1e-6)&(d>=peak_ratio*mx)).astype(np.uint8)
    n,lab=cv2.connectedComponents(peaks,8); centers=[]
    for i in range(1,n):
        ys,xs=np.where(lab==i)
        if len(xs): centers.append((float(xs.mean()),float(ys.mean()),float(d[ys[len(ys)//2],xs[len(xs)//2]])))
    kept=[]
    for x,y,_ in sorted(centers,key=lambda z:-z[2]):
        if all((x-a)**2+(y-b)**2>=min_peak_distance_px**2 for a,b in kept):kept.append((x,y))
    if len(kept)<2:return [comp],False
    markers=np.zeros_like(comp,np.int32)
    for k,(x,y) in enumerate(kept,1): markers[int(round(y)),int(round(x))]=k
    markers=cv2.dilate(markers.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(np.int32); markers[comp==0]=0
    ws=cv2.watershed(cv2.cvtColor(comp,cv2.COLOR_GRAY2BGR),markers)
    parts=[]
    for k in range(1,int(ws.max())+1):
        p=((ws==k)&(comp>0)).astype(np.uint8)*255
        if int((p>0).sum())>=10:parts.append(p)
    return (parts if len(parts)>=2 else [comp]),len(parts)>=2


def extract_particles(mask,scale_um_per_px=1.0,min_feret_um=10.0,separate=True,peak_ratio=0.70,peak_distance_px=8):
    n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8); base=[]
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=5: base.append((lab==i).astype(np.uint8)*255)
    if not base:return [],{'base_components':0,'splits':0}
    raw=[contour_measurements(p,scale_um_per_px) for p in base]; vals=[r['area_px'] for r in raw if r]; med=float(np.median(vals)) if vals else 0
    parts=[]; splits=0
    for p,r in zip(base,raw):
        suspicious=bool(r and (r['aspect_ratio']>1.8 or r['circularity']<0.55 or (med and r['area_px']>2.5*med)))
        if separate and suspicious:
            sp,did=_split_by_distance(p,peak_ratio,peak_distance_px)
            if did:
                ars=[int((q>0).sum()) for q in sp]
                if min(ars)>=max(10,0.10*np.median(ars)): parts.extend(sp); splits+=1
                else:parts.append(p)
            else:parts.append(p)
        else:parts.append(p)
    rows=[]; kept=[]
    for p in parts:
        m=contour_measurements(p,scale_um_per_px)
        if not m or m['feret_um']<min_feret_um:continue
        m['id']=len(rows)+1; m['spheroidal']=int(0.60<=m['roundness']<=1.0+1e-9); m['size_class']=size_class(m['feret_um'])
        rows.append(m); kept.append(p)
    return list(zip(kept,rows)),{'base_components':len(base),'splits':splits}


def _prepare_image(img,scale_um_per_px,max_dim=1800):
    h,w=img.shape[:2]
    if max(h,w)<=max_dim:return img,scale_um_per_px,1.0
    factor=max_dim/max(h,w); nw=max(1,int(round(w*factor))); nh=max(1,int(round(h*factor)))
    small=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    sc=scale_um_per_px/factor if scale_um_per_px else None
    return small,sc,factor


def _analyze_with_mask(img,mask,seg,scale,min_feret,separate,peak_ratio,peak_distance,original_shape=None):
    particles,split_meta=extract_particles(mask,scale,min_feret,separate,peak_ratio,peak_distance)
    area_px=float(mask.shape[0]*mask.shape[1]); included=float(sum(r['area_px'] for _,r in particles)); nod=float(sum(r['area_px'] for _,r in particles if r['spheroidal']))
    nodularity=100*nod/included if included else 0.0
    field_area=area_px*scale**2/1e6 if scale else None
    ferets=np.array([r['feret_um'] for _,r in particles],float); rounds=np.array([r['roundness'] for _,r in particles],float)
    return {'image':img,'mask':mask,'particles':particles,'rows':[r for _,r in particles],'nodularity':nodularity,'particle_count':len(particles),
            'particle_count_per_mm2':len(particles)/field_area if field_area else None,'field_area_mm2':field_area,
            'feret_min_um':float(ferets.min()) if len(ferets) else None,'feret_max_um':float(ferets.max()) if len(ferets) else None,
            'feret_mean_um':float(ferets.mean()) if len(ferets) else None,'feret_std_um':float(ferets.std(ddof=1)) if len(ferets)>1 else 0.0,
            'roundness_mean':float(rounds.mean()) if len(rounds) else 0.0,'graphite_area_fraction':100*included/area_px if area_px else 0.0,
            'segmentation':seg,'split_meta':split_meta,'size_counts':{str(k):int(sum(r['size_class']==k for _,r in particles)) for k in range(3,9)},
            'engine':seg.get('method'),'scale_um_per_px':scale,'min_feret_um':min_feret}


def analyze_field(img,engine='auto-stable',scale_um_per_px=None,min_feret_um=10.0,separate=True,peak_ratio=0.70,peak_distance_px=8,max_dim=1800):
    original=img
    work,scale,factor=_prepare_image(img,scale_um_per_px,max_dim)
    # Use pixel units internally when no physical calibration is supplied; the UI/report marks size-derived metrics as uncalibrated.
    internal_scale = float(scale) if scale is not None else 1.0
    candidates=[]
    methods={'iso-reference':segment_reference_otsu,'iso-enhanced':segment_enhanced_otsu,'blackhat':segment_blackhat}
    names=['iso-reference','iso-enhanced','blackhat'] if engine in ('auto-stable','auto-fast') else [engine]
    for name in names:
        mask,seg=methods[name](work,5)
        # Stability proxy: favor reasonable graphite coverage and avoid explosion of tiny components.
        frac,n,med=_mask_stats(mask)
        seg.update({'coverage':frac,'components':n,'median_component_px':med})
        if frac<0.001 or frac>0.35: score=-10
        else:
            score=0
            if 0.01<=frac<=0.15:score+=2
            if n<10000:score+=1
            if med>=20:score+=1
            # enhanced/reference get a small preference; blackhat is rescue candidate
            if name=='iso-enhanced':score+=0.3
        candidates.append((score,name,mask,seg))
    candidates.sort(key=lambda x:x[0],reverse=True)
    chosen=candidates[0]
    # For auto-stable, compare top two full results and choose the result whose nodularity is less sensitive to the segmentation method.
    evaluated=[]
    for score,name,mask,seg in candidates[:2] if engine in ('auto-stable','auto-fast') else candidates[:1]:
        f=_analyze_with_mask(work,mask,seg,internal_scale,min_feret_um,separate,peak_ratio,peak_distance_px,original.shape)
        f['_candidate_score']=score; f['_candidate_name']=name; evaluated.append(f)
    if engine in ('auto-stable','auto-fast') and len(evaluated)>1:
        # Prefer candidate with plausible coverage and fewer extreme tiny-particle fragments; nodularity is secondary.
        def q(f):
            r=f['rows']; tiny=sum(x['feret_um']<min_feret_um*1.15 for x in r); frag=tiny/max(1,len(r))
            return f['_candidate_score']-2*frag
        evaluated.sort(key=q,reverse=True)
    out=evaluated[0]
    out['analysis_scale_factor']=factor; out['original_shape']=original.shape[:2]; out['analysis_shape']=work.shape[:2]; out['physical_scale_supplied']=scale_um_per_px is not None
    out['speed_note']=f'Analysis resized to {work.shape[1]}×{work.shape[0]} for speed' if factor<1 else 'Full resolution analysis'
    return out


def aggregate_fields(fields):
    if not fields:return {}
    total=sum(sum(r['area_px'] for _,r in f['particles']) for f in fields); nod=sum(sum(r['area_px'] for _,r in f['particles'] if r['spheroidal']) for f in fields); count=sum(f['particle_count'] for f in fields)
    phys=[f['field_area_mm2'] for f in fields if f['field_area_mm2']]; total_mm2=sum(phys) if phys else None
    fer=[r['feret_um'] for f in fields for _,r in f['particles']]; rnd=[r['roundness'] for f in fields for _,r in f['particles']]; sz={str(k):sum(f['size_counts'].get(str(k),0) for f in fields) for k in range(3,9)}
    largest=max(fer) if fer else None
    return {'fields':len(fields),'particle_count':count,'nodularity':100*nod/total if total else 0.0,'particle_count_per_mm2':count/total_mm2 if total_mm2 else None,'total_area_mm2':total_mm2,
            'largest_feret_um':largest,'feret_mean_um':float(np.mean(fer)) if fer else None,'feret_std_um':float(np.std(fer,ddof=1)) if len(fer)>1 else 0.0,
            'roundness_mean':float(np.mean(rnd)) if rnd else 0.0,'size_counts':sz,'dominant_size':size_class(largest) if largest is not None else None,
            'minimum_fields_ok':len(fields)>=5,'minimum_particles_ok':count>=500,'full_sampling_requirement_ok':len(fields)>=5 and count>=500}


def overlay(img,field):
    out=img.copy()
    # If analysis was downscaled, resize masks/labels to original display resolution.
    sx=img.shape[1]/field['image'].shape[1]; sy=img.shape[0]/field['image'].shape[0]
    for mask,row in field['particles']:
        m=mask
        if m.shape[:2]!=img.shape[:2]: m=cv2.resize(m,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); color=(0,200,0) if row['spheroidal'] else (0,165,255)
        cv2.drawContours(out,cnts,-1,color,1)
        ys,xs=np.where(m>0)
        if len(xs):cv2.putText(out,str(row['id']),(int(xs.mean()),int(ys.mean())),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1,cv2.LINE_AA)
    return out
