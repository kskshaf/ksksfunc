from vapoursynth import core
import vapoursynth as vs
import re,math,functools,sys,os
import mvsfunc as mvf
import havsfunc as haf
import nnedi3_resample as nnrs
import vsTAAmbk as taa
import xvs
import muvsfunc as muf
from typing import Optional, Union
#GetY
def Y(clip):
	return core.std.ShufflePlanes(clip,0,vs.GRAY)
#GetU
def U(clip):
	return core.std.ShufflePlanes(clip,1,vs.GRAY)
#GetV
def V(clip):
	return core.std.ShufflePlanes(clip,2,vs.GRAY)
#GetPlane
def GetPlane(clip,n):
	return core.std.ShufflePlanes(clip,n,vs.GRAY)
###
GP=GetPlane
##BM3D_YUV444P16
def BM3D_444(clip,sigma,radius1,block_step,bm_range,ps_num,ps_range,rclip=None):
	src_opp = mvf.ToRGB(clip).bm3d.RGB2OPP(1)

	if rclip is None:
		rclip = src_opp
	else:
		rclip = mvf.ToRGB(rclip).bm3d.RGB2OPP(1)

	cpu = core.bm3dcpu.BM3Dv2(src_opp, rclip, sigma=sigma, block_step=block_step, bm_range=bm_range, radius=radius1, ps_num=ps_num, ps_range=ps_range, chroma=True)
	bm3d_f = mvf.ToYUV(core.bm3d.OPP2RGB(cpu,1),css=444,depth=16)
	return bm3d_f
##BM3D_YUV420P16
def BM3DCPU(clip,sigma,radius1,block_step,bm_range,ps_num,ps_range,rclip=None,outdepth=16):
	clip=clip.fmtc.bitdepth(bits=32)
	if rclip is None:
		rclip = clip
	else:
		rclip = mvf.Depth(rclip,32)

	bm3d = core.bm3dcpu.BM3Dv2(clip, rclip, sigma=sigma, block_step=block_step, bm_range=bm_range, radius=radius1, ps_num=ps_num, ps_range=ps_range, chroma=False).fmtc.bitdepth(bits=outdepth)
	return bm3d
##CropPart
##用以裁剪出一处矩形区域
def CropPart(src,src_crop,l=0,t=0,r=0,b=0):
	w=src.width
	h=src.height
	left = core.std.Crop(src,right=w-l)
	right = core.std.Crop(src,left=w-r)
	top = core.std.Crop(src,left=l,right=r,bottom=h-t)
	bottom = core.std.Crop(src,left=l,right=r,top=h-b)
	part = core.std.Crop(src_crop,left=l,right=r,top=t,bottom=b)
	top_mid_bottom = core.std.StackVertical([top,part,bottom])
	right_last = core.std.StackHorizontal([left,top_mid_bottom,right])
	return right_last
##Adaptive Denoise Mask
##亮度自适应降噪（使用指数函数）
#x1,x2 for light strength
#y1,y2 for denoise (mask)strength
def Adaptive_DenoiseMask(clip,x1,y1,x2,y2):
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('You must input a "clip"!')
    if not isinstance(x1, int):
        raise ValueError('"x1" must be a int!')
    if not isinstance(y1, int):
        raise ValueError('"y1" must be a int!')
    if not isinstance(x2, int):
        raise ValueError('"x2" must be a int!')
    if not isinstance(y2, int):
        raise ValueError('"y2" must be a int!')
    elif x1 < x2:
        raise ValueError('"x1" must bigger than "x2"!')
    elif y1 < y2:
        raise ValueError('"y1" must bigger than "y2"!')
    else:
       a_expr=(y1-y2)/(x1**2-x2**2)
       b1=y1-a_expr*x1**2
       b2=y2-a_expr*x2**2
       
    if b1!=b2:
       raise ValueError('Calculate error!')
    else:
        b_expr=b2
        clip=core.std.Expr(clip,f'x {x1} > {y1} x {x2} < {y2} x 2 pow {a_expr} * {b_expr} + ? ?')
    return clip
##Adaptive Denoise
### br:MinBlur range ; rgm:The mode of "rgvs.RemoveGrain()"
def Adaptive_Denoise(input,sigma,radius1,block_step,bm_range,ps_num,ps_range,x1=52428,y1=65535,x2=32678,y2=1200,blur=True,br=1,rgm=20,showmask=False):
	get_y = Y(input)

	if blur:
		blur = haf.MinBlur(get_y,br).rgvs.RemoveGrain(rgm)
	else:
		blur = get_y

	bm3d_a = BM3DCPU(input,sigma=sigma,radius1=radius1,block_step=block_step,bm_range=bm_range,ps_num=ps_num,ps_range=ps_range)
	mask_w = Adaptive_DenoiseMask(blur,x1=x1,y1=y1,x2=x2,y2=y2)
	denoised = core.std.MaskedMerge(input,bm3d_a,mask_w)
	if showmask:
		output = mask_w
	else:
		output = denoised
	return output
###
ADN = Adaptive_Denoise
ADNM = Adaptive_DenoiseMask
#20220927
#NormalDeband
def Deband(denoised,nrmask,range1=6,range2=10,y1=48,c1=36,r1=36,y2=36,c2=24,r2=24,thr=0.6,thrc=0.5,elast=2.0,first_plane=True):
    deband=core.f3kdb.Deband(denoised,range1,y1,c1,r1,0,0)
    deband=core.f3kdb.Deband(deband,range2,y2,c2,r2,0,0)
    deband=mvf.LimitFilter(deband,denoised,thr=thr,thrc=thrc,elast=elast)
    deband=core.std.MaskedMerge(deband,denoised,nrmask,first_plane)
    return deband
#DebandMask(LoliHouse)
def DBMask(clip):
    nr8=mvf.Depth(clip,8)
    luma   = core.std.ShufflePlanes(nr8, 0, vs.YUV).resize.Bilinear(format=vs.YUV420P8)
    nrmasks = core.tcanny.TCanny(nr8,sigma=0.8,op=2,gmmax=255,mode=1,planes=[0,1,2]).std.Expr(["x 7 < 0 65535 ?",""],vs.YUV420P16)
    nrmaskb = core.tcanny.TCanny(nr8,sigma=1.3,t_h=6.5,op=2,planes=0)
    nrmaskg = core.tcanny.TCanny(nr8,sigma=1.1,t_h=5.0,op=2,planes=0)
    nrmask  = core.std.Expr([nrmaskg,nrmaskb,nrmasks,nr8],["a 20 < 65535 a 48 < x 256 * a 96 < y 256 * z ? ? ?",""],vs.YUV420P16)
    nrmask  = core.std.Maximum(nrmask,0).std.Maximum(0).std.Minimum(0)
    nrmask  = core.rgvs.RemoveGrain(nrmask,[20,0])
    return nrmask
#AdaptiveSharp(TCanny&MinBlur)
def Sharp(src_nos,sigma=1,t_h=12,t_l=2.5,mode=1,op=1,br=1,rmode=0,thr=0.6,thrc=0.5,elast=10,showmask=False):
    get_y=Y(src_nos)
    edge=core.tcanny.TCanny(get_y,sigma=sigma,t_h=t_h,t_l=t_l,mode=mode,op=op).std.Maximum().std.Minimum().std.Maximum()
    edge=haf.MinBlur(edge,br).rgvs.RemoveGrain(rmode)
    mmd_s=core.std.MergeDiff(src_nos,core.std.MakeDiff(src_nos,haf.MinBlur(src_nos,br)))
    sharp=mvf.LimitFilter(mmd_s,src_nos,thr=thr,thrc=thrc,elast=elast)
    sharp=core.std.MaskedMerge(mmd_s,src_nos,edge)
    if showmask:
        sharp_out=edge
    else:
        sharp_out=sharp
    return sharp_out
#20221016

#Dering(https://www.skyey2.com/forum.php?mod=viewthread&tid=32112&extra=page%3D1)
def ContraDering(clip,ecstrength=15,ecrmode=18,conn=[1,2,1,2,4,2,1,2,1],mrad=1.0,mthr=115,csrange=1.0,rpmode=23):
    dr=haf.EdgeCleaner(clip,strength=ecstrength,rmode=ecrmode).std.Convolution(conn)
    mask_r=haf.HQDeringmod(clip,show=True,mrad=mrad,mthr=mthr)
    dr=core.std.MaskedMerge(clip,dr,mask_r)
    cp=haf.ContraSharpening(dr,clip,csrange)
    rp=core.rgvs.Repair(dr,cp,rpmode)
    return rp
###Modifde of xyx98's rescalef.
#Dering&AA after rescale
#GRAY output
def rescalef_aa(src: vs.VideoNode,kernel: str,w=None,h=None,bh=None,bw=None,mask=True,mask_dif_pix=2,show="result",rsmode="znedi3",dering=False,ecstrength=15,ecrmode=18,conn=[1,2,1,2,4,2,1,2,1],mrad=1.0,mthr=115,csrange=1.0,rpmode=23,not_aa=True,aatype=1,aarepair=-20,sharp=-0.5,mtype=3,postaa=True,stablize=2,aacycle=1,thin=0,dark=0.15,aamask=0,postfilter_descaled=None,selective=False,upper=0.0001,lower=0.00001,**args):
    #for decimal resolution descale,refer to GetFnative
    if src.format.color_family is not vs.GRAY:
        src=Y(src)

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if bh is None:
        bh=1080

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")
    
    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")
    ####
    src=mvf.Depth(src,16)
    luma=src
    cargs=xvs.cropping_args(src.width,src.height,h,bh,bw)
    ####
    if kernel in ["Debilinear","Despline16","Despline36","Despline64"]:
        luma_de=eval("core.descale.{k}(luma.fmtc.bitdepth(bits=32),**cargs.descale_gen())".format(k=kernel))
        luma_up=eval("core.resize.{k}(luma_de,**cargs.resize_gen())".format(k=kernel[2:].capitalize()))
    elif kernel=="Debicubic":
        luma_de=core.descale.Debicubic(luma.fmtc.bitdepth(bits=32),b=args.get("b"),c=args.get("c"),**cargs.descale_gen())
        luma_up=core.resize.Bicubic(luma_de,filter_param_a=args.get("b"),filter_param_b=args.get("c"),**cargs.resize_gen())
    else:
        luma_de=core.descale.Delanczos(luma.fmtc.bitdepth(bits=32),taps=args.get("taps"),**cargs.descale_gen())
        luma_up=core.resize.Lanczos(luma_de,filter_param_a=args.get("taps"),**cargs.resize_gen())
    
    diff = core.std.Expr([luma.fmtc.bitdepth(bits=32), luma_up], f'x y - abs dup 0.015 > swap 0 ?').std.Crop(10, 10, 10, 10).std.PlaneStats()

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        luma_de=postfilter_descaled(luma_de)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=args.get("pscrn")
    exp=args.get("exp")

    luma_rescale=nnrs.nnedi3_resample(luma_de,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=rsmode,**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)

    if dering:
        luma_rsdr_aa=ContraDering(luma_rescale,ecstrength=ecstrength,ecrmode=ecrmode,conn=conn,mrad=mrad,mthr=mthr,csrange=csrange,rpmode=rpmode)
    else:
        luma_rsdr_aa=luma_rescale
    
    if not_aa:
        luma_rsdr_aa=luma_rsdr_aa
    else:
        luma_rsdr_aa=taa.TAAmbk(luma_rsdr_aa,aatype=aatype,aarepair=aarepair,sharp=sharp,mtype=mtype,postaa=postaa,stablize=stablize,cycle=aacycle,thin=thin,dark=dark,showmask=aamask)#TAA

    def calc(n,f): 
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]
        return fout

    luma_rescale=core.std.ModifyFrame(luma_rescale,[diff,luma_rescale],calc)

    if mask:
        mask=core.std.Expr([luma,luma_up.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        mask=xvs.expand(mask,cycle=2)
        mask=xvs.inpand(mask,cycle=2)

        luma_rescale=core.std.MaskedMerge(luma_rescale,luma,mask)
    
    if selective:
        base=upper-lower
        #x:rescale y:src
        expr=f"x.diff {upper} > y x.diff {lower} < x {upper} x.diff -  {base} / y * x.diff {lower} - {base} / x * + ? ?"
        luma_rescale=core.akarin.Expr([luma_rescale,luma], expr)

    if show=="descale":
        return luma_de
    elif show=="dering":
        return luma_rsdr_aa
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask
    elif show=="diff":
        return core.text.FrameProps(luma_rescale,"diff", scale=2)
    else:
        return luma_rsdr_aa
#20221030
#MRcoref(rescalef) of xvs.
#Only GRAY input and output.
def MRcoref(clip:vs.VideoNode,kernel:str,w=None,h=None,bh=None,bw=None,mask: Union[bool,vs.VideoNode]=True,mask_dif_pix:float=2,postfilter_descaled=None,mthr:list[int]=[2,2],taps:int=3,b:float=0,c:float=0.5,multiple:float=1,maskpp=None,show:str="result",blur_mask=False,ds_aa=False,aatype=1,aarepair=-20,sharp=-0.5,mtype=3,postaa=True,stablize=2,aacycle=0,thin=0,dark=0.15,aamask=0,dehalo=False,dering=False,thlimi=60,thlima=150,brightstr=0.4,darkstr=0,drthr=2,**args):
    if clip.format.color_family != vs.GRAY or clip.format.bits_per_sample != 16:
        raise ValueError("input clip should be GRAY16!")

    src_h,src_w=clip.height,clip.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")

    if bh is None:
        bh=src_h

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")
    
    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")

    #clip=core.fmtc.bitdepth(clip,bits=16)
    #luma=Y(clip)
    #src_w,src_h=clip.width,clip.height
    cargs=xvs.cropping_args(src_w,src_h,h,bh,bw)
    clip32=core.fmtc.bitdepth(clip,bits=32)
    kernel=kernel[2:]
    descaled=core.descale.Descale(clip32,kernel=kernel.lower(),taps=taps,b=b,c=c,**cargs.descale_gen())
    upscaled=xvs.resize_core(kernel.capitalize(),taps,b,c)(descaled,**cargs.resize_gen())
    diff=core.std.Expr([clip32,upscaled],"x y - abs dup 0.015 > swap 0 ?").std.Crop(10, 10, 10, 10).std.PlaneStats()
    def calc(n,f): 
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]*multiple
        return fout

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        descaled=postfilter_descaled(descaled)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")
    etype=args.get("etype")
    pscrn=args.get("pscrn")
    exp=args.get("exp")
    sigmoid=args.get("sigmoid")

    if ds_aa:
        descaled_aa=taa.TAAmbk(descaled.fmtc.bitdepth(bits=16),aatype=aatype,aarepair=aarepair,sharp=sharp,mtype=mtype,postaa=postaa,stablize=stablize,cycle=aacycle,thin=thin,dark=dark,showmask=aamask) #TAA
        rescale=nnrs.nnedi3_resample(descaled_aa,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid,mode="znedi3",**cargs.nnrs_gen())
    else:
        rescale=nnrs.nnedi3_resample(descaled,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid,mode="znedi3",**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)
        
    if dehalo:
        rescale=haf.FineDehalo(rescale,thlimi=thlimi,thlima=thlima,brightstr=brightstr,darkstr=darkstr)

    if dering:
        rescale=muf.mdering(rescale,thr=drthr)

#rescale=nnrs.nnedi3_resample(descaled_aa,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid,mode="znedi3",**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)

    if mask is True:
        mask=core.std.Expr([clip,upscaled.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(maskpp):
            mask=maskpp(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])
    if blur_mask is True:
        mask=mask.rgvs.RemoveGrain(20)

        rescale=core.std.MaskedMerge(rescale,clip,mask)
    elif isinstance(mask,vs.VideoNode):
        if mask.width!=src_w or mask.height!=src_h or mask.format.color_family!=vs.GRAY:
            raise ValueError("mask should have same resolution as source,and should be GRAY")
        mask=core.fmtc.bitdepth(mask,bits=16,dmode=1)
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    else:
        mask=core.std.BlankClip(rescale)

    if show.lower()=="result":
        return rescale
    elif show.lower()=="mask" and mask:
        return mask
    elif show.lower()=="descale":
        return descaled #after postfilter_descaled
#20230208

