import test from "node:test";
import assert from "node:assert/strict";

import { normalizeAnswer, normalizeArticle } from "../src/normalize/normalizer.js";

const context = {
  userToken: "huang-wei-yan-30",
  crawlTime: "2026-04-15T12:00:00.000Z",
  markdownEnabled: true
};

test("normalizeAnswer maps answer fields into unified schema", () => {
  const answer = {
    id: "123",
    type: "answer",
    content: "<p>Answer body</p>",
    created_time: 1775719819,
    updated_time: 1775725196,
    comment_count: 5,
    question: { id: "q1", title: "Question title" },
    author: { name: "Alice", url_token: "alice" },
    reaction: { statistics: { favorites: 9, like_count: 2 } }
  };
  const out = normalizeAnswer(answer, context);
  assert.equal(out.source_type, "answer");
  assert.equal(out.source_id, "123");
  assert.equal(out.question_id, "q1");
  assert.equal(out.url, "https://www.zhihu.com/question/q1/answer/123");
  assert.equal(out.content_text, "Answer body");
  assert.equal(out.favorite_count, 9);
});

test("normalizeArticle maps article fields into unified schema", () => {
  const article = {
    id: "a1",
    type: "article",
    title: "Article title",
    content: "<h2>T</h2><p>Body</p>",
    excerpt: "Body",
    created: 1775202911,
    updated: 1775324491,
    comment_count: 7,
    voteup_count: 13,
    url: "http://zhuanlan.zhihu.com/p/a1",
    author: { name: "Bob", url_token: "bob" },
    reaction: { statistics: { favorites: 11, like_count: 3 } }
  };
  const out = normalizeArticle(article, context);
  assert.equal(out.source_type, "article");
  assert.equal(out.source_id, "a1");
  assert.equal(out.title, "Article title");
  assert.equal(out.question_id, null);
  assert.equal(out.url, "http://zhuanlan.zhihu.com/p/a1");
  assert.equal(out.voteup_count, 13);
  assert.match(out.content_markdown, /^## T/m);
});
