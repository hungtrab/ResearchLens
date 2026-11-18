"""Exercise the reader in Chromium against the app in-process, without a listening port."""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
from playwright.sync_api import expect, sync_playwright
from researchlens.api import create_app

ROOT = Path(__file__).resolve().parents[1]
SEED = "W2184957013"


def main():
    with (
        TemporaryDirectory() as work,
        TestClient(
            create_app(data_dir=ROOT / "data", db_path=Path(work) / "state.db")
        ) as api,
        sync_playwright() as pw,
    ):

        def dispatch(route):
            req = route.request
            url = urlsplit(req.url)
            if url.hostname != "researchlens.test" or url.path.endswith("/pdf"):
                route.abort()
                return
            response = api.request(
                req.method,
                url.path + ("?" + url.query if url.query else ""),
                content=req.post_data_buffer,
                headers={
                    "content-type": req.headers.get("content-type", "application/json")
                },
            )
            route.fulfill(
                status=response.status_code,
                body=response.content,
                content_type=response.headers.get(
                    "content-type", "application/octet-stream"
                ),
            )

        browser = pw.chromium.launch(args=["--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1350, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/*", dispatch)
        page.goto("http://researchlens.test/#reading?paper_id=" + SEED)
        panel = page.locator("#passage-finder")
        panel.locator("summary").click()
        radio = panel.get_by_role("radio")
        radio.nth(5).check()
        action = panel.get_by_role("button", name="Find similar papers")
        action.click()
        hits = panel.locator(".passage-results article")
        expect(hits).to_have_count(3)

        def check_quotes(selected):
            items = api.get(
                f"/api/papers/{SEED}/passage-matches?sentence={selected}"
            ).json()["items"]
            for i, item in enumerate(items):
                expect(hits.nth(i).locator("blockquote")).to_have_text(item["quote"])
                assert (
                    item["quote"]
                    in api.get("/api/papers/" + item["id"]).json()["abstract"]
                )
            return items

        check_quotes(5)
        radio.nth(2).check()
        expect(hits).to_have_count(0)
        action.click()
        expect(hits).to_have_count(3)
        items = check_quotes(2)
        panel.screenshot(path=ROOT / "artifacts/screenshots/passage_reader.png")
        first_ids = {item["id"] for item in items}
        hits.first.get_by_role("button", name="Find more like this paper").click()
        follow = panel.locator(".passage-followup article")
        expect(follow).to_have_count(3)

        def check_page(after):
            response = api.get(
                f"/api/papers/{SEED}/passage-matches",
                params={"sentence": 2, "after": after},
            )
            assert response.status_code == 200
            matches = response.json()["items"]
            assert not first_ids.intersection(item["id"] for item in matches)
            for i, item in enumerate(matches):
                expect(follow.nth(i).locator(".paper-title")).to_have_text(
                    item["title"]
                )
                expect(follow.nth(i).locator("blockquote")).to_have_text(item["quote"])

        check_page(items[0]["id"])
        panel.get_by_role("button", name="Show next three").click()
        expect(
            panel.get_by_role("heading", name="Next in the original list")
        ).to_be_visible()
        check_page("next")
        hits.first.get_by_role("button", name=items[0]["title"], exact=True).click()
        expect(page.locator(".reader-heading h1")).to_have_text(items[0]["title"])
        assert not errors, errors
        browser.close()
        print(
            "Reader verified: two selected sentences, continuation and feedback pages, source excerpts, first result opens."
        )


if __name__ == "__main__":
    main()
