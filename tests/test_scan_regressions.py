"""Regression cases from the September handoff audit."""
from pathlib import Path
import cv2
import numpy as np
import pytest
import pymupdf
from spejl.detect.ocr import Detection, RapidOcrBackend
from spejl.detect.rotations import _resolve_ambiguous_tilts
from spejl.models import Axis
from spejl.qa.verify import verify_mirror, check_rendered_text, render_diff_overlay
from spejl.image_io import write_image
from spejl.vector.pdf_mirror import mirror_pdf


class Fixed:
    def __init__(self, detections, shape=(200, 300)):
        self.detections, self.shape = detections, shape

    def detect_and_recognise(self, image):
        return self.detections if image.shape[:2] == self.shape else []


def det(text, box, confidence=.99):
    x0, y0, x1, y1 = box
    return Detection(text, ((x0,y0),(x1,y0),(x1,y1),(x0,y1)), confidence)


def test_verify_does_not_discard_pixels_outside_a_text_box(tmp_path):
    source = np.full((200,300,3),255,np.uint8)
    output = source.copy()
    output[85,100:281] = 0
    src, dst = tmp_path/'source.png', tmp_path/'output.png'
    write_image(src,source)
    write_image(dst,output)
    report = verify_mirror(src,dst,Axis.VERTICAL,Fixed([det('1234',(80,70,140,100))]))
    assert not report.passed
    assert report.outside_text_px == 81  # half-open exclusion x=140:240


@pytest.mark.parametrize('reading,confidence', [('24',.99),('42',.60)])
def test_rendered_wording_is_not_certified_on_disagreement_or_uncertainty(reading,confidence):
    image=np.full((200,300,3),255,np.uint8)
    issues=check_rendered_text(image,[('42',(20,50,100,80))],
                              Fixed([det(reading,(20,50,100,80),confidence)]))
    assert len(issues)==1


def test_missing_rendered_wording_is_reported():
    assert check_rendered_text(np.zeros((200,300,3),np.uint8),
                               [('42',(20,50,100,80))],Fixed([]))


def test_rendered_wording_requires_one_to_one_matches():
    issues=check_rendered_text(np.zeros((200,300,3),np.uint8),
        [('42',(20,50,100,80)),('42',(21,50,101,80))],Fixed([det('42',(20,50,100,80))]))
    assert len(issues)==1


def test_image_write_reports_missing_parent(tmp_path):
    with pytest.raises(OSError,match='Could not write'):
        write_image(tmp_path/'absent'/'output.png',np.zeros((20,20,3),np.uint8))


def test_overlay_write_reports_failure(tmp_path,monkeypatch):
    src=tmp_path/'source.png'
    write_image(src,np.zeros((20,20,3),np.uint8))
    monkeypatch.setattr(cv2,'imwrite',lambda *args:False)
    with pytest.raises(OSError,match='Could not write'):
        render_diff_overlay(src,src,Axis.VERTICAL,tmp_path/'overlay.png')


@pytest.mark.parametrize('text,quad,angle',[
    ('5461',((503.,832.),(568.,825.),(571.,855.),(507.,862.)),6.14662565964667),
    ('4381',((842.,1103.),(875.,1099.),(883.,1164.),(850.,1168.)),97.01650174472291),
])
def test_t25_drawing_corroborates_isolated_dimension_slope(text,quad,angle):
    image=cv2.imread(str(Path(__file__).parent/'data'/'t25.jpg'))
    assert image is not None
    original=Detection(text,quad,.99,angle)
    assert _resolve_ambiguous_tilts([original],image)[0].angle_deg == angle


def test_isolated_tilt_without_drawing_support_still_snaps():
    d=Detection('5461',((30.,30.),(90.,24.),(93.,54.),(33.,60.)),.99,6.)
    assert _resolve_ambiguous_tilts([d],np.full((200,300,3),255,np.uint8))[0].angle_deg==0


@pytest.mark.parametrize('mixed',[False,True])
def test_pdf_images_survive_including_mixed_pages(tmp_path,mixed):
    # A large asymmetric shape exercises content preservation without relying
    # on OCR recognising artificial text. Text-aware raster calls stay real.
    image=np.full((500,500,3),255,np.uint8)
    image[50:400,40:100]=0
    png=tmp_path/'embedded.png'
    write_image(png,image)
    src=tmp_path/'source.pdf'
    with pymupdf.open() as pdf:
        p=pdf.new_page(width=120,height=120)
        p.insert_image(p.rect,filename=str(png))
        if mixed:
            p.draw_rect(pymupdf.Rect(10,100,90,110),fill=(0,0,0))
        pdf.save(src)
    dst=tmp_path/'mirrored.pdf'
    result=mirror_pdf(src,dst)
    with pymupdf.open(src) as pdf:
        pix=pdf[0].get_pixmap(dpi=300,alpha=False)
        source=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    with pymupdf.open(dst) as pdf:
        pix=pdf[0].get_pixmap(dpi=300,alpha=False)
        output=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    assert np.mean(np.abs(cv2.flip(source,1).astype(float)-output)) < 2
    assert any(f.code=='pdf-rasterized' for f in result.pages[0].flags)
    assert not any(f.code=='image-removed' for f in result.pages[0].flags)


def test_uncertain_ocr_retry_uses_new_pixels_and_preserves_geometry():
    backend=RapidOcrBackend.__new__(RapidOcrBackend)
    quad=((20,20),(60,20),(60,40),(20,40))
    calls=[]
    def engine(image):
        calls.append(image.shape)
        if len(calls)==1: return [(quad,'A70',.71)],None
        return [(((24,24),(104,24),(104,64),(24,64)),'870',.77)],None
    backend._ocr=engine
    result=backend.detect_and_recognise(np.zeros((100,100,3),np.uint8))
    assert result[0].text=='870'
    assert result[0].conf==.71
    assert result[0].quad==quad
    assert len(calls)==2


def test_uncertain_ocr_retry_cannot_replace_a_read_with_a_fragment():
    backend=RapidOcrBackend.__new__(RapidOcrBackend)
    quad=((20,20),(60,20),(60,40),(20,40))
    calls=[]
    def engine(image):
        calls.append(1)
        return [(quad,'A70' if len(calls)==1 else '70',.71 if len(calls)==1 else .99)],None
    backend._ocr=engine
    assert backend.detect_and_recognise(np.zeros((100,100,3),np.uint8))[0].text=='A70'


def test_verify_supports_the_pipelines_exact_twofold_upscale(tmp_path):
    src, dst=tmp_path/'src.png',tmp_path/'dst.png'
    image=np.full((100,150,3),255,np.uint8)
    image[10:80,20:25]=0
    write_image(src,image)
    write_image(dst,cv2.flip(cv2.resize(image,None,fx=2,fy=2,
                                     interpolation=cv2.INTER_LANCZOS4),1))
    report=verify_mirror(src,dst,Axis.VERTICAL,Fixed([]))
    assert report.passed
    assert report.differing_px==0


def test_pdf_scanned_text_is_readable_after_mirroring(tmp_path):
    from spejl.qa import fixture_gen
    from spejl.detect.rotations import detect_all_orientations
    info=fixture_gen.generate(tmp_path/'fixture',dpi=150)
    image=cv2.imread(str(info['png']))
    h,w=image.shape[:2]
    src,dst=tmp_path/'scan.pdf',tmp_path/'scan_mirrored.pdf'
    with pymupdf.open() as pdf:
        page=pdf.new_page(width=w*72/150,height=h*72/150)
        page.insert_image(page.rect,filename=str(info['png']))
        pdf.save(src)
    mirror_pdf(src,dst)
    with pymupdf.open(dst) as pdf:
        pix=pdf[0].get_pixmap(dpi=150,alpha=False)
        image=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    reads=detect_all_orientations(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),RapidOcrBackend())
    assert any(d.text=='Stue' for d in reads)


def test_verify_reports_wrong_wording_even_with_clean_geometry(tmp_path):
    image=np.full((200,300,3),255,np.uint8)
    src,dst=tmp_path/'src.png',tmp_path/'dst.png'
    write_image(src,image)
    write_image(dst,image)
    class Pair:
        calls=0
        def detect_and_recognise(self,image):
            if image.shape[:2] != (200,300): return []
            self.calls+=1
            return [det('42',(20,65,95,115))] if self.calls==1 else [det('24',(205,65,280,115))]
    report=verify_mirror(src,dst,Axis.VERTICAL,Pair())
    assert not report.passed
    assert report.outside_text_px==0
    assert report.text_issues


def test_raster_pipeline_never_reports_success_when_write_fails(tmp_path):
    from spejl.raster.pipeline import mirror_raster
    src=tmp_path/'src.png'
    write_image(src,np.full((200,300,3),255,np.uint8))
    dst=tmp_path/'absent'/'dst.png'
    with pytest.raises(OSError,match='Could not write'):
        mirror_raster(src,dst,backend=Fixed([]))
    assert not dst.exists()
