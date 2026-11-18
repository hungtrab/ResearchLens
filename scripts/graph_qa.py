"""Interaction regression: drag transmits force, pins correctly and cools to rest."""
import json, os, math
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
BASE=os.environ.get('RESEARCHLENS_BASE_URL','http://127.0.0.1:8765')
with sync_playwright() as pw:
    binary=Path(pw.chromium.executable_path)
    if not binary.exists(): binary=sorted((Path.home()/'.cache/ms-playwright').glob('chromium-*/chrome-linux*/chrome'))[-1]
    browser=pw.chromium.launch(executable_path=str(binary),headless=True,args=['--disable-gpu','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+'/#graph');page.locator('#network .node').first.wait_for()
    def mount():
        page.evaluate('''() => {
          document.querySelector('#main').innerHTML='<div class="graph-wrap"><svg id="physics-test" style="width:800px;height:500px"></svg></div>';
          const nodes=['A','B','C','D'].map((id,i)=>({id,title:'Test node '+id,x:-.75+i*.5,y:0,pagerank:.25,community:0}));
          mountCitationGraph(document.querySelector('#physics-test'),{nodes,edges:[{source:'A',target:'B'},{source:'B',target:'C'}]},800,500);
        }''')
    def coords():
        return page.locator('#physics-test .graph-node').evaluate_all('els => Object.fromEntries(els.map(el => {const m=el.transform.baseVal.consolidate().matrix;return [el.dataset.node,[m.e,m.f]];}))')
    def dist(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])
    def drag():
        before=coords();box=page.locator('[data-node=A] .node').bounding_box();x=box['x']+box['width']/2;y=box['y']+box['height']/2
        page.mouse.move(x,y);page.mouse.down()
        for step in range(1,11):
            page.mouse.move(x+step*11,y+step*7);page.wait_for_timeout(35)
        page.wait_for_timeout(300);during=coords()
        assert dist(before['B'],during['B'])>4,('Connected neighbor did not follow drag',before,during)
        assert dist(before['C'],during['C'])>1,('Force did not reach second-hop node',before,during)
        now=page.locator('[data-node=A] .node').bounding_box()
        assert math.hypot(now['x']+now['width']/2-(x+110),now['y']+now['height']/2-(y+70))<1,'Dragged node lost pointer pin'
        page.mouse.up();return before,during
    mount();before,during=drag()
    expect(page.locator('#physics-test')).to_have_attribute('data-simulating','false',timeout=7000)
    rest=coords();page.wait_for_timeout(250);assert coords()==rest,'Graph kept moving after cooling'
    assert dist(during['B'],rest['B'])>.5,'Release did not settle the cluster'
    page.get_by_role('button',name='Zoom in',exact=True).click();assert 'scale(1.25)' in page.locator('.graph-layer').get_attribute('transform')
    page.get_by_role('button',name='Reset graph view',exact=True).click();assert coords()==before
    # At non-unit zoom, drag still follows the pointer and moves linked nodes.
    page.get_by_role('button',name='Zoom out',exact=True).click();drag()
    expect(page.locator('#physics-test')).to_have_attribute('data-simulating','false',timeout=7000)
    # Reduced motion keeps direct manipulation coupled, without post-release inertia.
    page.emulate_media(reduced_motion='reduce');mount();drag()
    expect(page.locator('#physics-test')).to_have_attribute('data-simulating','false')
    rest=coords();page.wait_for_timeout(250);assert coords()==rest
    # Removing a graph while it is hot cancels work on the next animation frame.
    page.emulate_media(reduced_motion='no-preference');mount();drag();page.evaluate("() => document.querySelector('#main').replaceChildren()")
    page.wait_for_timeout(200);assert not errors,errors
    result={'passed':True,'javascript_errors':errors,'base_url':BASE,'neighbor_displacement_during_drag':dist(before['B'],during['B']),'checks':['direct and second-hop neighbors move','dragged node stays pinned','release settles and stops','zoomed drag uses correct coordinates','reset restores layout','reduced motion has no inertia','detached graph stops']}
    (ROOT/'artifacts/graph_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));browser.close()
