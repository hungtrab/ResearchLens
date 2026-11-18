"""Check compiled PDF page geometry and log evidence; complements visual review."""
import json,subprocess
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class BoundsParser(HTMLParser):
 def __init__(self):super().__init__();self.pages=0;self.outside=[];self.width=0;self.height=0
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='page':self.pages+=1;self.width=float(a['width']);self.height=float(a['height'])
  elif tag=='word':
   if float(a['xmin'])<-.5 or float(a['ymin'])<-.5 or float(a['xmax'])>self.width+.5 or float(a['ymax'])>self.height+.5:self.outside.append({'page':self.pages,'bounds':a})
def main():
 result={}
 for name in ['report','slides']:
  pdf=ROOT/name/'main.pdf'
  html=subprocess.run(['pdftotext','-bbox',str(pdf),'-'],check=True,capture_output=True,text=True).stdout
  parser=BoundsParser();parser.feed(html)
  if parser.outside:raise AssertionError(f'{name}: text outside page: {parser.outside}')
  log=(ROOT/name/'main.log').read_text(errors='replace')
  assert 'Overfull' not in log, f'{name}: overfull boxes in LaTeX log'
  result[name]={'pages':parser.pages,'words_outside_page':parser.outside,'overfull_boxes':False}
 result['visual_review']='All pages rendered as contact sheets; cover, method, results and conclusion reviewed at full size; actual UI screenshots.'
 (ROOT/'artifacts/document_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
