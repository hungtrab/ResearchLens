"""Browser acceptance checks for real Projects, PDF reading, notes and graph interactions."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
BASE=os.environ.get('RESEARCHLENS_BASE_URL','http://127.0.0.1:8765').rstrip('/')
OUT=ROOT/'artifacts/screenshots';OUT.mkdir(exist_ok=True)
with sync_playwright() as pw:
    executable=Path(pw.chromium.executable_path)
    if not executable.exists(): executable=sorted((Path.home()/'.cache/ms-playwright').glob('chromium-*/chrome-linux*/chrome'))[-1]
    browser=pw.chromium.launch(executable_path=str(executable),headless=True,args=['--disable-gpu','--disable-dev-shm-usage'])
    context=browser.new_context(viewport={'width':1600,'height':1100})
    page=context.new_page();errors=[];page.on('pageerror',lambda err:errors.append(str(err)))
    page.goto(BASE+'/#projects');page.locator('#empty-new-project').click()
    page.locator('#example-brief').click();page.locator('#project-form [type=submit]').click()
    page.locator('[data-attach]').first.wait_for();project_id=page.url.split('id=')[1]
    for _ in range(3):
        page.locator('[data-attach]').first.click();page.locator('[data-attach]').first.wait_for()
    expect(page.locator('.project-paper-row')).to_have_count(3)
    expect(page.locator('#project-network .node').first).to_be_visible()
    expect(page.locator('#toast')).not_to_have_class('show',timeout=6000);page.wait_for_timeout(900);page.screenshot(path=str(OUT/'projects.png'))
    # Real public PDF, no response substitution for the PDF or this acceptance flow.
    pid='W3007332492'
    page.goto(BASE+'/#reading?paper_id='+pid)
    frame=page.locator('#paper-pdf');expect(frame).to_be_visible(timeout=45000)
    expect(page.locator('#document-label')).to_have_text('Original PDF')
    assert frame.get_attribute('src').startswith('blob:')
    page.locator('[name=status]').select_option('reading');page.locator('[name=page]').fill('4')
    notes='Compare benchmark protocols and reproducibility across graph neural network datasets. Track the evaluation splits before citing performance claims.'
    page.locator('[name=notes]').fill(notes);page.locator('#reading-form [type=submit]').click()
    expect(page.locator('#note-status')).to_have_text('Saved')
    page.locator('#jump-page').click();assert '#page=4' in frame.get_attribute('src')
    page.wait_for_timeout(1200);page.screenshot(path=str(OUT/'reading.png'))
    page.reload();expect(page.locator('[name=notes]')).to_have_value(notes);expect(page.locator('[name=page]')).to_have_value('4')
    # UTF-8 notes exceed fetch keepalive's 64 KiB limit, but normal autosave must work.
    large='ệ'*22000
    page.locator('[name=notes]').fill(large);page.locator('#reading-form [type=submit]').click();expect(page.locator('#note-status')).to_have_text('Saved')
    assert context.request.get(BASE+'/api/papers/'+pid+'/reading').json()['notes']==large
    page.locator('[name=notes]').fill(notes);expect(page.locator('#note-status')).to_have_text('Saved')
    # Recover a locally retained draft after a failed save and route change.
    def fail_save(route):
        if route.request.method=='PUT':route.fulfill(status=503,content_type='application/json',body='{"detail":"Simulated unavailable storage"}')
        else:route.continue_()
    context.route('**/api/papers/*/reading',fail_save)
    draft='Unsaved recovery check: preserve this reading observation.'
    page.locator('[name=notes]').fill(draft);expect(page.locator('#note-status')).to_have_text('Not saved')
    page.locator('nav [data-view=projects]').click();page.locator('.project-card').wait_for()
    context.unroute('**/api/papers/*/reading',fail_save)
    page.goto(BASE+'/#reading?paper_id='+pid);expect(page.locator('[name=notes]')).to_have_value(draft)
    expect(page.locator('#note-status')).to_have_text('Saved');assert context.request.get(BASE+'/api/papers/'+pid+'/reading').json()['notes']==draft
    page.locator('[name=notes]').fill(notes);expect(page.locator('#note-status')).to_have_text('Saved')
    # Honest missing-PDF fallback using another actual corpus paper.
    # Find a real missing-PDF record independently; IDs are corpus-specific.
    papers=[json.loads(x) for x in (ROOT/'data/papers.jsonl').read_text().split('\n') if x.strip()]
    missing=next(x['id'] for x in papers if not x.get('pdf_url'))
    page.goto(BASE+'/#reading?paper_id='+missing);expect(page.locator('.pdf-notice')).to_be_visible();expect(page.locator('.abstract-document')).to_contain_text('NOT THE FULL PAPER')
    page.goto(BASE+'/#graph');page.locator('#network .node').first.wait_for();page.wait_for_timeout(700)
    node=page.locator('#network .node').first;node.click();expect(page.locator('.graph-selection [data-read]')).to_be_visible()
    page.get_by_role('button',name='Zoom in',exact=True).click();assert 'scale(1.25)' in page.locator('#network .graph-layer').get_attribute('transform')
    page.get_by_role('button',name='Reset graph view',exact=True).click()
    page.locator('[data-groups]').check();page.locator('[data-labels]').check();page.locator('[data-labels]').uncheck()
    page.wait_for_timeout(300);page.screenshot(path=str(OUT/'obsidian-network.png'))
    page.get_by_role('searchbox',name='Find a node').fill('zzzz-no-match');assert page.locator('#network .graph-node:not(.dimmed)').count()==0
    page.get_by_role('searchbox',name='Find a node').fill('')
    # Mobile project / reader and reduced motion.
    page.set_viewport_size({'width':390,'height':844});page.goto(BASE+'/#projects?id='+project_id)
    page.locator('#project-network .node').first.wait_for();assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.wait_for_timeout(700);page.screenshot(path=str(OUT/'projects-mobile.png'),full_page=True)
    page.goto(BASE+'/#reading?paper_id='+missing);page.locator('.pdf-notice').wait_for();assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.wait_for_timeout(300);page.screenshot(path=str(OUT/'reading-mobile.png'),full_page=True)
    page.emulate_media(reduced_motion='reduce');assert page.locator('.reader-heading').evaluate('(el)=>getComputedStyle(el).animationName')=='none'
    assert not errors,errors
    (ROOT/'artifacts/workspace_qa.json').write_text(json.dumps({'passed':True,'javascript_errors':errors,'real_pdf_paper':pid,'flows':['project create','brief suggestions','paper membership','project graph','real public PDF','page bookmark','UTF-8 notes over64KiB','autosave and reload','failed save draft recovery','missing PDF fallback','graph selection and zoom','graph groups and labels','graph title search','mobile overflow','reduced motion']},indent=2))
    context.request.delete(BASE+'/api/projects/'+project_id)
    context.close();browser.close()
