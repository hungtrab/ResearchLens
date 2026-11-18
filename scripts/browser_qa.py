"""Capture real UI states and assert the end-to-end demo workflow."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
BASE=os.environ.get('RESEARCHLENS_BASE_URL','http://127.0.0.1:8765').rstrip('/')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/screenshots';OUT.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    chromium=Path(p.chromium.executable_path)
    if not chromium.exists():
        candidates=sorted((Path.home()/'.cache/ms-playwright').glob('chromium-*/chrome-linux*/chrome'))
        if not candidates: raise RuntimeError('Install browser: uv run playwright install chromium')
        chromium=candidates[-1]
    browser=p.chromium.launch(executable_path=str(chromium) if chromium.exists() else None,headless=True,args=['--disable-gpu','--disable-dev-shm-usage'])
    context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page=context.new_page();errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+'/#search?q=graph%20neural%20networks')
    page.locator('.paper-card').first.wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'discover.png'))
    page.locator('.save-button').first.click();expect(page.locator('#saved-count')).to_have_text('1');expect(page.locator('#toast')).not_to_have_class('show',timeout=6000)
    page.locator('.paper-title').first.click();page.locator('#paper-body h2').wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'paper.png'))
    page.locator('#close-dialog').click()
    page.locator('nav a[data-view="graph"]').click();page.locator('#network .node').first.wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'network.png'))
    page.locator('nav a[data-view="reviews"]').click();page.locator('#opinion-form button').click();page.locator('#analysis-result .evidence').first.wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'reviews.png'))
    page.locator('nav a[data-view="library"]').click();page.locator('#recommendations .paper-card').first.wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'library.png'))
    with page.expect_download() as download_info:page.locator('a[download]').click()
    download=download_info.value;assert download.suggested_filename=='researchlens-library.bib'
    page.locator('nav a[data-view="analytics"]').click();page.locator('.stat-strip').wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'analytics.png'))
    page.set_viewport_size({'width':390,'height':844});page.goto(BASE+'/#search?q=deep%20learning');page.locator('.paper-card').first.wait_for();page.wait_for_timeout(300);page.screenshot(path=str(OUT/'mobile.png'))
    overflow=page.evaluate('document.documentElement.scrollWidth > innerWidth');assert not overflow,'Mobile horizontal overflow'
    assert not errors,errors
    (ROOT/'artifacts/browser_qa.json').write_text(json.dumps({'passed':True,'javascript_errors':errors,'mobile_overflow':overflow,'screenshots':[p.name for p in OUT.glob('*.png')],'flows':['search','save','detail','graph','opinion','recommendations','bibtex export','analytics','mobile']},indent=2))
    context.close();browser.close()
