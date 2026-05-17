import test from "node:test";
import assert from "node:assert/strict";

import { extractImageUrls, htmlToMarkdown, htmlToText } from "../src/transform/html.js";

test("extractImageUrls returns unique urls in document order", () => {
  const html =
    '<p>hello</p><img src="https://a/img1.png"><img src="https://a/img1.png"><img data-original="https://a/img2.jpg">';
  const urls = extractImageUrls(html);
  assert.deepEqual(urls, ["https://a/img1.png", "https://a/img2.jpg"]);
});

test("htmlToText strips tags and preserves readable spacing", () => {
  const html = "<p>Hello <strong>World</strong></p><p>Line 2</p>";
  assert.equal(htmlToText(html), "Hello World\n\nLine 2");
});

test("htmlToMarkdown keeps links and headings in lightweight markdown", () => {
  const html = '<h2>Title</h2><p>A <a href="https://x.test">link</a></p>';
  const md = htmlToMarkdown(html);
  assert.match(md, /^## Title/m);
  assert.match(md, /\[link\]\(https:\/\/x\.test\)/);
});
