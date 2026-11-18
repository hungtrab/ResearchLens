"""Bounded access to public repository PDFs. No arbitrary URL proxy or paywall bypass."""
from urllib.parse import urlsplit, urljoin, unquote
import time
import httpx

PDF_HOSTS=frozenset({'arxiv.org','export.arxiv.org','openreview.net','proceedings.mlr.press','aclanthology.org','www.aclweb.org','papers.nips.cc','proceedings.neurips.cc','openaccess.thecvf.com','proceedings.iclr.cc'})
MAX_PDF_BYTES=20*1024*1024

def supported_pdf(url):
    if not isinstance(url,str) or len(url)>2000: return False
    try:
        p=urlsplit(url)
        path=unquote(p.path)
        raw_pdf=p.hostname=='raw.githubusercontent.com' and path.startswith('/mlresearch/') and path.endswith('.pdf') and not any(part in ('.','..') for part in path.split('/')) and '%' not in path and '\\' not in path
        return p.scheme=='https' and (p.hostname in PDF_HOSTS or raw_pdf) and p.port in (None,443) and not p.username and not p.password and not p.fragment and not any(c in url for c in ('\\','\r','\n','\t'))
    except ValueError: return False

def download_pdf(url):
    if not supported_pdf(url): raise ValueError('This source cannot be embedded. Open the original source to read the paper.')
    deadline=time.monotonic()+25
    try:
        with httpx.Client(timeout=httpx.Timeout(12,connect=8),follow_redirects=False,trust_env=False,headers={'User-Agent':'ResearchLens/1.1 (academic paper reader)','Accept':'application/pdf'}) as client:
            for _ in range(4):
                if not supported_pdf(url): raise ValueError('The PDF redirects outside supported public repositories. Open the original source.')
                with client.stream('GET',url) as response:
                    if response.is_redirect:
                        url=urljoin(url,response.headers.get('location','')); continue
                    response.raise_for_status()
                    if int(response.headers.get('content-length','0'))>MAX_PDF_BYTES: raise ValueError('PDF exceeds the 20 MB reader limit. Open the original source.')
                    chunks=[]; size=0
                    for chunk in response.iter_bytes(chunk_size=65536):
                        size+=len(chunk)
                        if size>MAX_PDF_BYTES: raise ValueError('PDF exceeds the 20 MB reader limit. Open the original source.')
                        if time.monotonic()>deadline: raise OSError('PDF source took too long. Try again or open the original source.')
                        chunks.append(chunk)
                    content=b''.join(chunks)
                    if not content.lstrip().startswith(b'%PDF-'): raise OSError('The publisher did not return a PDF. Open the original source.')
                    return content
            raise OSError('Too many publisher redirects. Open the original source.')
    except httpx.HTTPError as exc: raise OSError('PDF source is currently unavailable. Try again or open the original source.') from exc
