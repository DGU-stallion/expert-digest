import { extractImageUrls, htmlToMarkdown, htmlToText } from "../transform/html.js";

function toIsoSeconds(epochSeconds) {
  if (typeof epochSeconds !== "number" || Number.isNaN(epochSeconds)) {
    return null;
  }
  return new Date(epochSeconds * 1000).toISOString();
}

function toCount(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return 0;
}

function normalizeCommon({
  sourceType,
  sourceId,
  authorName,
  authorToken,
  title,
  questionId,
  questionTitle,
  url,
  createdAt,
  updatedAt,
  commentCount,
  voteupCount,
  favoriteCount,
  likeCount,
  contentHtml,
  crawlTime,
  markdownEnabled
}) {
  const html = String(contentHtml ?? "");
  return {
    source_type: sourceType,
    source_id: String(sourceId ?? ""),
    author_name: authorName ?? null,
    author_token: authorToken ?? null,
    title: title ?? null,
    question_id: questionId ?? null,
    question_title: questionTitle ?? null,
    url: url ?? null,
    created_at: createdAt ?? null,
    updated_at: updatedAt ?? null,
    comment_count: toCount(commentCount),
    voteup_count: toCount(voteupCount),
    favorite_count: toCount(favoriteCount),
    like_count: toCount(likeCount),
    content_html: html,
    content_markdown: markdownEnabled ? htmlToMarkdown(html) : null,
    content_text: htmlToText(html),
    image_urls: extractImageUrls(html),
    crawl_time: crawlTime,
    warnings: []
  };
}

export function normalizeAnswer(answer, context) {
  const id = String(answer?.id ?? "");
  const questionId = answer?.question?.id != null ? String(answer.question.id) : null;
  const title = answer?.question?.title ?? null;
  const url = questionId ? `https://www.zhihu.com/question/${questionId}/answer/${id}` : null;

  return normalizeCommon({
    sourceType: "answer",
    sourceId: id,
    authorName: answer?.author?.name ?? null,
    authorToken: answer?.author?.url_token ?? context?.userToken ?? null,
    title,
    questionId,
    questionTitle: title,
    url,
    createdAt: toIsoSeconds(answer?.created_time),
    updatedAt: toIsoSeconds(answer?.updated_time),
    commentCount: answer?.comment_count,
    voteupCount: answer?.voteup_count,
    favoriteCount: answer?.reaction?.statistics?.favorites,
    likeCount: answer?.reaction?.statistics?.like_count,
    contentHtml: answer?.content ?? "",
    crawlTime: context?.crawlTime ?? new Date().toISOString(),
    markdownEnabled: Boolean(context?.markdownEnabled)
  });
}

export function normalizeArticle(article, context) {
  const id = String(article?.id ?? "");
  const url = article?.url ?? `https://zhuanlan.zhihu.com/p/${id}`;
  return normalizeCommon({
    sourceType: "article",
    sourceId: id,
    authorName: article?.author?.name ?? null,
    authorToken: article?.author?.url_token ?? context?.userToken ?? null,
    title: article?.title ?? null,
    questionId: null,
    questionTitle: null,
    url,
    createdAt: toIsoSeconds(article?.created),
    updatedAt: toIsoSeconds(article?.updated),
    commentCount: article?.comment_count,
    voteupCount: article?.voteup_count,
    favoriteCount: article?.reaction?.statistics?.favorites,
    likeCount: article?.reaction?.statistics?.like_count,
    contentHtml: article?.content ?? "",
    crawlTime: context?.crawlTime ?? new Date().toISOString(),
    markdownEnabled: Boolean(context?.markdownEnabled)
  });
}
