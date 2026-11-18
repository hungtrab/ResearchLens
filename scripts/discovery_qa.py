"""Live browser acceptance for the conference index and online import workflow.
Run against a disposable database: this intentionally imports one real paper.
"""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
BASE=os.environ.get('RESEARCHLENS_BASE_URL','http://127.0.0.1:8767').rstrip('/')
OUT=ROOT/'artifacts/screenshots';OUT.mkdir(exist_ok=True)
with sync_playwright() as pw:
    executable=Path(pw.chromium.executable_path)
    if not executable.exists():executable=sorted((Path.home()/'.cache/ms-playwright').glob('chromium-*/chrome-linux*/chrome'))[-1]
    browser=pw.chromium.launch(executable_path=str(executable),headless=True,args=['--disable-gpu','--disable-dev-shm-usage'])
    context=browser.new_context(viewport={'width':1600,'height':1100})
    page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    status=context.request.get(BASE+'/api/discovery/status').json()
    assert status['papers']>100000 and status['years']==[2020,2026]
    page.goto(BASE+'/#catalog?q=graph%20neural%20networks&conference=ICML&year_min=2025&year_max=2025')
    page.locator('.discovery-card').first.wait_for()
    assert page.locator('.discovery-card .year').all_text_contents()==['2025']*page.locator('.discovery-card').count()
    page.screenshot(path=str(OUT/'conferences.png'))
    page.locator('.coverage-panel summary').click();expect(page.locator('.coverage-panel tbody tr')).to_have_count(8)
    page.screenshot(path=str(OUT/'conference_coverage.png'))
    page.locator('#discovery-next').click();expect(page.locator('.pagination')).to_contain_text('21–40')
    page.goto(BASE+'/#catalog?q=zzResearchLensNoSuchPaperzz');expect(page.locator('#discovery-results')).to_contain_text('No matching papers found.')
    expect(page.locator('#discovery-results a[href^="#online"]')).to_be_visible()
    page.goto(BASE+'/#online?q=Flow%20Matching%20for%20Generative%20Modeling')
    page.locator('.discovery-card').first.wait_for(timeout=45000)
    page.screenshot(path=str(OUT/'online_search.png'))
    button=page.locator('[data-import-paper]').first;pid=button.get_attribute('data-import-paper')
    title=button.locator('xpath=ancestor::article').locator('h3').inner_text()
    button.click();page.locator('#reading-form').wait_for(timeout=45000)
    expect(page.locator('#main')).to_contain_text(title)
    paper=context.request.get(BASE+'/api/papers/'+pid);assert paper.status==200
    page.locator('[name=notes]').fill('Track this paper from live discovery to the project reading list.')
    expect(page.locator('#note-status')).to_have_text('Saved',timeout=10000)
    page.reload();expect(page.locator('[name=notes]')).to_have_value('Track this paper from live discovery to the project reading list.')
    project=context.request.post(BASE+'/api/projects',data={'title':'Online discovery acceptance','brief':'Flow matching and generative models'}).json()
    attached=context.request.post(BASE+'/api/projects/'+project['id']+'/papers/'+pid);assert attached.ok
    graph=context.request.get(BASE+'/api/projects/'+project['id']+'/graph').json();assert any(n['id']==pid for n in graph['nodes'])
    page.goto(BASE+'/#online?q=Flow%20Matching%20for%20Generative%20Modeling');page.locator('.discovery-card').first.wait_for(timeout=45000)
    expect(page.locator('.discovery-card').filter(has_text=title).first).to_contain_text('IN WORKSPACE')
    page.set_viewport_size({'width':390,'height':844});page.goto(BASE+'/#catalog?q=graph');page.locator('.discovery-card').first.wait_for()
    page.locator('.coverage-panel summary').click()
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'),'Mobile overflow'
    page.screenshot(path=str(OUT/'conference_mobile.png'))
    assert not errors,errors
    result={'passed':True,'catalog_papers':status['papers'],'years':status['years'],'imported_id':pid,'imported_title':title,'javascript_errors':errors,'live_provider':'OpenAlex','flows':['conference BM25','conference/year filters','coverage matrix','pagination','empty-result online link','live online search','import and read','notes reload','attach imported paper to project','project graph membership','duplicate badge','mobile layout']}
    (ROOT/'artifacts/discovery_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    context.request.delete(BASE+'/api/projects/'+project['id'])
    context.close();browser.close()
