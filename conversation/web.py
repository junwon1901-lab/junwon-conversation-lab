"""Bounded public-web retrieval. Every entry point checks the request's ON/OFF policy."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import ipaddress
import re
import socket
from urllib.parse import urlsplit
import requests
from bs4 import BeautifulSoup


class OfflineError(RuntimeError):
    pass


@dataclass(frozen=True)
class NetworkPolicy:
    enabled: bool = False

    def require(self):
        if not self.enabled:
            raise OfflineError('인터넷 OFF: 외부 요청을 보내지 않았습니다.')


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def public_url(url, check_dns=True):
    """Only public HTTP(S), no credentials or arbitrary ports."""
    try:
        p = urlsplit(url)
        if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
            return False
        if p.port not in (None, 80, 443) or p.hostname.lower() == 'localhost' or p.hostname.lower().endswith(('.local', '.internal', '.localhost')):
            return False
        try:
            if not ipaddress.ip_address(p.hostname).is_global:
                return False
        except ValueError:
            pass
        if not check_dns:
            return True
        addresses = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == 'https' else 80))
        return bool(addresses) and all(ipaddress.ip_address(a[4][0]).is_global for a in addresses)
    except (ValueError, OSError):
        return False


def excerpt(url, policy):
    policy.require()
    if not public_url(url):
        return ''
    # Redirects are deliberately not followed, avoiding an unchecked redirect to a private host.
    with requests.get(url, timeout=(3, 5), allow_redirects=False, stream=True,
                      headers={'User-Agent': 'JunwonConversationLab/1.0 (educational reader)'}) as response:
        if response.status_code != 200 or 'text/html' not in response.headers.get('content-type', ''):
            return ''
        data = bytearray()
        for chunk in response.iter_content(8192):
            data.extend(chunk)
            if len(data) >= 180_000:
                break
        soup = BeautifulSoup(bytes(data), 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'form']):
            tag.decompose()
        return re.sub(r'\s+', ' ', ' '.join(p.get_text(' ', strip=True) for p in soup.select('main p, article p, p')))[:2000]


def probe(policy):
    policy.require()
    checks = []
    for service, url in [('모델 저장소', 'https://huggingface.co/api/models/Qwen/Qwen3-1.7B'),
                         ('공개 웹', 'https://www.wikipedia.org/')]:
        try:
            response = requests.get(url, timeout=(3, 4))
            checks.append({'service': service, 'ok': response.status_code == 200, 'status': response.status_code})
        except requests.RequestException:
            checks.append({'service': service, 'ok': False, 'status': 'timeout_or_network_error'})
    connected = any(c['ok'] for c in checks)
    return {'connected': connected, 'checked_at': now(), 'checks': checks,
            'detail': ' · '.join(c['service'] + (' 응답 확인' if c['ok'] else ' 확인 실패') for c in checks)}


def search(query, policy, read_pages=True):
    policy.require()  # BEFORE constructing DDGS or making DNS/HTTP requests.
    from ddgs import DDGS
    query = re.sub(r'\s+', ' ', str(query)).strip()[:180]
    if not query:
        raise ValueError('검색할 내용을 입력하세요.')
    # DDGS selects currently enabled providers; disabled engines must not be hard-coded.
    results = DDGS(timeout=8).text(query, region='us-en', safesearch='moderate', max_results=3, backend='auto')
    sources, seen = [], set()
    for row in results:
        url = str(row.get('href', ''))
        if url in seen or not public_url(url, check_dns=False):
            continue
        seen.add(url)
        body = ''
        if read_pages and len(sources) < 2:
            try:
                body = excerpt(url, policy)
            except (requests.RequestException, OSError, ValueError):
                pass
        sources.append({'title': str(row.get('title', '검색 결과'))[:180], 'url': url,
                        'text': (body or str(row.get('body', '')))[:2000],
                        'kind': '본문 일부' if body else '검색 요약', 'retrieved_at': now()})
        if len(sources) == 3:
            break
    return sources
