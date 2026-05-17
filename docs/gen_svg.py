"""
Expert Digest PPT — SVG Generator
Generates all 20 SVG slides for ppt-master conversion.
"""
import os, math

OUT = r"D:\Project\ppt-master-2.6.0\projects\expert_digest_ppt169_20260514\svg_final"
IMG = r"D:\Project\ppt-master-2.6.0\projects\expert_digest_ppt169_20260514\images"
os.makedirs(OUT, exist_ok=True)

W, H = 1280, 720
ML, MR, MT, MB = 60, 60, 50, 50
CW = W - ML - MR  # content width = 1160

# ─── Color palette ───
C_BG     = '#0F172A'
C_CARD   = '#1E293B'
C_BORDER = '#334155'
C_BLUE   = '#3B82F6'
C_PURPLE = '#8B5CF6'
C_GREEN  = '#10B981'
C_ORANGE = '#F59E0B'
C_RED    = '#F87171'
C_TEXT   = '#E2E8F0'
C_SUB    = '#94A3B8'
C_MUTED  = '#64748B'
C_CODE   = '#0B1120'

_GRAD_ID = 0
def _grad():
    global _GRAD_ID; _GRAD_ID += 1
    gid = f"g{_GRAD_ID}"
    return gid, f'''<linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" stop-color="{C_BLUE}"/>
  <stop offset="100%" stop-color="{C_PURPLE}"/>
</linearGradient>'''

# ─── SVG Helpers ───

def esc(text):
    """Escape XML special characters."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def header(title, page, total=20):
    """Return SVG defs + background + title bar."""
    gid, grad = _grad()
    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    {grad}
    <radialGradient id="bg" cx="85%" cy="15%" r="45%">
      <stop offset="0%" stop-color="{C_BLUE}" stop-opacity="0.08"/>
      <stop offset="100%" stop-color="{C_BLUE}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="{C_BG}"/>
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect x="{ML}" y="{MT}" width="4" height="40" rx="2" fill="url(#{gid})"/>
  <text x="{ML+20}" y="{MT+28}" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="32" font-weight="bold" fill="{C_TEXT}">{title}</text>
  <text x="{W-MR}" y="{H-30}" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="10" fill="{C_MUTED}" text-anchor="end">{page:02d} / {total:02d}</text>'''

def footer(body):
    return f'{body}\n</svg>'

def rect(x, y, w, h, fill=C_CARD, stroke=C_BORDER, r=12, sw=1, dash=None, opacity=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    op_attr = f' fill-opacity="{opacity}"' if opacity else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}{op_attr}/>'

def txt(x, y, text, size=18, color=C_TEXT, bold=False, anchor='start', family="'Segoe UI','Microsoft YaHei',sans-serif", spacing=0):
    """Single-line text. If spacing > 0, returns multiple tspans."""
    bold_a = ' font-weight="bold"' if bold else ''
    safe = esc(text)
    if spacing > 0:
        lines = [f'<tspan x="{x}" dy="{spacing if i>0 else 0}">{esc(line)}</tspan>' for i, line in enumerate(text)]
        return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}"{bold_a} text-anchor="{anchor}">{"".join(lines)}</text>'
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}"{bold_a} text-anchor="{anchor}">{safe}</text>'

def tspan(parent_x, y, segments, size=18, family="'Segoe UI','Microsoft YaHei',sans-serif"):
    """Text with differently colored segments. segments = [(text, color, bold), ...]"""
    parts = []
    for text, color, bold in segments:
        b = ' font-weight="bold"' if bold else ''
        parts.append(f'<tspan fill="{color}"{b}>{esc(text)}</tspan>')
    return f'<text x="{parent_x}" y="{y}" font-family="{family}" font-size="{size}">{"".join(parts)}</text>'

def card(x, y, w, h, fill=C_CARD, stroke=C_BORDER):
    return rect(x, y, w, h, fill, stroke)

def card_title(x, y, text, color=C_BLUE, size=20):
    return txt(x+24, y+35, text, size, color, bold=True)

def bullet(x, y, text, size=16, color=C_TEXT, indent=0):
    ix = x + 24 + indent
    return txt(ix, y, f'• {text}', size, color)

def note(x, y, text, size=14, color=C_SUB, indent=0):
    ix = x + 24 + indent
    return txt(ix, y, text, size, color)

def image_tag(x, y, w, h, path):
    """Embed an image — use relative path from svg_final to images."""
    # path is absolute like D:\Project\...\images\file.png
    # We need a path relative to svg_final: ../images/file.png
    fname = os.path.basename(path)
    return f'''<image x="{x}" y="{y}" width="{w}" height="{h}" href="../images/{fname}"/>
<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" stroke="{C_BORDER}" stroke-width="1"/>'''


# ═══════════════════════════════════════════════
# PAGE GENERATORS
# ═══════════════════════════════════════════════

def page01():
    """Cover"""
    gid, grad = _grad()
    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    {grad}
    <radialGradient id="bg" cx="75%" cy="20%" r="50%">
      <stop offset="0%" stop-color="{C_BLUE}" stop-opacity="0.1"/>
      <stop offset="100%" stop-color="{C_BLUE}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="{C_BG}"/>
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect x="440" y="240" width="400" height="4" rx="2" fill="url(#{gid})"/>
  <text x="640" y="320" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="56" font-weight="bold" fill="{C_TEXT}" text-anchor="middle">Expert Digest</text>
  <text x="640" y="375" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="26" fill="url(#{gid})" text-anchor="middle">专家内容知识蒸馏引擎</text>
  <text x="640" y="425" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="18" fill="{C_SUB}" text-anchor="middle">将领域专家的公开智慧转化为 AI Agent 可直接使用的知识资产</text>
  <rect x="520" y="460" width="240" height="2" rx="1" fill="{C_BORDER}"/>
  <text x="640" y="520" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="16" fill="{C_MUTED}" text-anchor="middle">项目结项汇报 · 2026年5月</text>
</svg>'''

def page02():
    """Background — two columns"""
    xl, xr = ML, ML + 600
    cw = 540
    body = header('项目背景与意义', 2)
    # Left: pain points
    body += card(xl, 120, cw, 480)
    body += card_title(xl, 120, '核心痛点', C_RED)
    pts = [
        '专家知识分散在各平台，无法系统化沉淀',
        '内容散落在知乎、公众号等平台，缺乏统一的管理工具',
        'AI 无法真正学会专家的思考方式',
        '现有方案仅做关键词检索和匹配，缺乏对思维框架的理解',
        'RAG 方案仅做表层检索，回答缺乏深度',
        '无法捕捉专家的决策逻辑和独特思维模式',
    ]
    for i, p in enumerate(pts[:3]):
        body += bullet(xl, 190 + i*80, p, 17, C_TEXT)
        body += note(xl, 215 + i*80, pts[i+3] if i+3 < len(pts) else '', 14, C_SUB)

    # Right: solutions
    body += card(xr, 120, cw, 480)
    body += card_title(xr, 120, '项目目标', C_GREEN)
    sols = [
        ('构建端到端知识处理管线', '从爬取 → 清洗 → 向量化 → 聚类 → 知识蒸馏，全流程自动化'),
        ('Handbook 结构化学习手册', '56页深度内容，7大章节，涵盖完整的投资认知框架'),
        ('Skill Agent 技能描述文件', '18KB Agent专用技能，可直接用于 Claude Code 等 AI IDE'),
        ('Wiki Vault 双向链接知识体系', '60+ 概念条目，每个概念均可追溯到原文出处'),
    ]
    for i, (s, d) in enumerate(sols):
        body += bullet(xr, 190 + i*90, s, 17, C_TEXT)
        body += note(xr, 215 + i*90, d, 14, C_SUB)
    return footer(body)

def page03():
    """Architecture — image + subsystem cards"""
    body = header('整体架构总览', 3)
    body += image_tag(140, 115, 1000, 350, f'{IMG}\\p03_architecture.png')
    xl, xr = ML, ML+600
    body += card(xl, 490, 540, 110)
    body += card_title(xl, 490, '数据获取层 · Zhihu Crawler', C_BLUE, 18)
    body += note(xl, 525, '知乎内容爬取，基于浏览器上下文的创新方案', 15)
    body += note(xl, 550, '反爬规避、速率限制、断点续爬', 15)
    body += card(xr, 490, 540, 110)
    body += card_title(xr, 490, '知识蒸馏层 · Expert Digest', C_PURPLE, 18)
    body += note(xr, 525, '存储 · 处理 · 分析 · 蒸馏 · 输出', 15)
    body += note(xr, 550, 'LangGraph 编排 + 零依赖向量化', 15)
    return footer(body)

def page04():
    """Crawler intro — three info cards"""
    x1, x2 = ML, ML+400
    body = header('数据来源 — Zhihu Crawler', 4)
    body += card(x1, 120, 360, 200)
    body += card_title(x1, 120, '项目概览', C_BLUE, 18)
    body += bullet(x1, 175, 'Zhihu Crawler v0.1.0', 16)
    body += note(x1, 200, 'Node.js + 浏览器上下文', 14)
    body += note(x1, 230, '目标用户：黄彦臻', 14)
    body += note(x1, 260, '数据量：824篇内容', 14)

    body += card(x2, 120, 360, 200)
    body += card_title(x2, 120, '核心挑战', C_ORANGE, 18)
    body += bullet(x2, 175, '知乎反爬机制严密', 16)
    body += note(x2, 200, '裸请求直接返回 403/401', 14, C_SUB, indent=20)
    body += note(x2, 230, '需要处理签名算法和', 14, C_SUB, indent=20)
    body += note(x2, 255, '风控检测机制', 14, C_SUB, indent=20)

    body += card(x2+400, 120, 360, 200)
    body += card_title(x2+400, 120, 'API 调研', C_GREEN, 18)
    body += note(x2+400, 175, '/members/{token}/answers', 13, C_SUB, indent=20)
    body += note(x2+400, 200, 'offset=0&limit=10&sort_by=created', 13, C_SUB, indent=20)
    body += note(x2+400, 230, '/members/{token}/articles', 13, C_SUB, indent=20)
    body += note(x2+400, 260, '含 content, comment_count 等字段', 13, C_SUB, indent=20)

    # Solution
    body += card(x1, 360, 760, 270)
    body += card_title(x1, 360, '创新解决方案：浏览器上下文方案', C_PURPLE, 18)
    body += bullet(x1, 420, '复用 Chrome 已登录态，无需手工导出 Cookie', 16)
    body += note(x1, 448, '利用 opencli 建立浏览器桥接，在页面内执行 fetch 请求', 14, C_SUB, indent=20)
    body += bullet(x1, 490, '避免复现知乎签名算法和风控细节', 16)
    body += note(x1, 518, '站在巨人肩膀上，不重复造轮子', 14, C_SUB, indent=20)
    body += bullet(x1, 560, '保守反爬策略保证稳定性', 16)
    body += note(x1, 588, '低速率 + 随机抖动 · 遇风险信号立即停止 · 最小接口集', 14, C_SUB, indent=20)

    return footer(body)

def page05():
    """Crawler architecture"""
    body = header('爬虫技术架构', 5)
    body += image_tag(190, 115, 900, 280, f'{IMG}\\p05_crawler.png')

    modules = [
        ('运行原理', C_BLUE, [
            '基于 opencli 浏览器上下文',
            '复用 Chrome 登录知乎会话',
            '页面内 fetch 调用 API',
        ]),
        ('数据流设计', C_PURPLE, [
            '双层输出：Raw + Normalized',
            'Raw层：原始响应，可追溯审计',
            'Normalized：标准化知识库',
        ]),
        ('反爬与容错', C_ORANGE, [
            '默认 2.5s 请求间隔 + 随机抖动',
            '401/403/429 立即停止',
            '断点续爬 + ID 去重机制',
        ]),
    ]
    for i, (title, color, items) in enumerate(modules):
        x = ML + i*395
        body += card(x, 420, 365, 210)
        body += card_title(x, 420, title, color, 17)
        for j, item in enumerate(items):
            body += note(x, 485+j*45, f'• {item}', 15, C_TEXT)

    return footer(body)

def page06():
    """Data output — raw/normalized"""
    body = header('数据输出 — 双层设计', 6)
    xl, xr = ML, ML+600
    body += card(xl, 120, 540, 220)
    body += card_title(xl, 120, 'Raw 层 · 原始数据', C_BLUE, 18)
    body += note(xl, 170, '• 完整的 API 原始响应，数据零损失', 15)
    body += note(xl, 200, '• 可追溯、可审计', 15)
    body += note(xl, 230, '• 支持重新处理和验证', 15)
    body += note(xl, 270, '输出：answers.jsonl / articles.jsonl', 14, C_MUTED, 24)

    body += card(xr, 120, 540, 220)
    body += card_title(xr, 120, 'Normalized 层 · 标准化数据', C_PURPLE, 18)
    body += note(xr, 170, '• 统一 Schema，知识库直接可用', 15)
    body += note(xr, 200, '• HTML / Markdown / Text 三格式', 15)
    body += note(xr, 230, '• 包含完整的元数据信息', 15)
    body += note(xr, 270, '输出：index/content_index.jsonl', 14, C_MUTED, 24)

    # Field details
    body += card(xl, 370, 1160, 260)
    body += card_title(xl, 370, '核心字段设计', C_GREEN, 18)
    fields = [
        ('source_type / source_id', '唯一标识，区分 answer 和 article'),
        ('author_name / author_token', '作者信息，便于归属和溯源'),
        ('content_html / content_markdown / content_text', '三格式内容覆盖所有使用场景'),
        ('created_at / updated_at', '时间戳，支持时序分析和增量更新'),
        ('voteup_count / comment_count', '互动数据，衡量内容质量和热度'),
    ]
    for i, (f, d) in enumerate(fields):
        y = 430 + i*42
        body += tspan(xl+24, y, [(f'  {f}', C_TEXT, True), (f'  —  {d}', C_SUB, False)], 14, "Consolas,'Courier New',monospace")

    return footer(body)

def page07():
    """Pipeline — three stages"""
    body = header('数据处理流水线', 7)
    stages = [
        ('① 数据导入', C_BLUE, [
            '三种来源统一接入',
            'JSONL / Markdown / 知乎导出',
            '清洗：去重、规范化',
            '质量过滤层',
        ]),
        ('② 向量化 ✨创新点', C_PURPLE, [
            '哈希词袋 (Bag-of-Words)',
            '零外部依赖，纯本地计算',
            '结果可复现、确定性算法',
            'API 成本 = 0！',
        ]),
        ('③ SQLite 存储', C_GREEN, [
            '单文件数据库，便携部署',
            'documents 表：原始文档',
            'chunks 表：智能分块',
            'embeddings 表：向量存储',
            'topics / communities 表',
        ]),
    ]
    for i, (title, color, items) in enumerate(stages):
        x = ML + i*400
        body += card(x, 120, 360, 400)
        body += card_title(x, 120, title, color, 18)
        for j, item in enumerate(items):
            body += note(x, 185+j*50, f'• {item}', 15, C_TEXT)

    # Arrow connectors between columns
    for i in range(2):
        cx1 = ML + (i+1)*400 - 20
        cx2 = cx1 + 40
        body += f'<line x1="{cx1}" y1="320" x2="{cx2}" y2="320" stroke="{C_BORDER}" stroke-width="2" marker-end="url(#arrow)"/>'
    body += '''<defs>
  <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#64748B"/>
  </marker>
</defs>'''

    # Bottom note
    body += note(ML, 550, '整个处理流程无需调用任何外部 API，完全本地化运行，数据安全可控。', 15, C_SUB)
    return footer(body)

def page08():
    """Clustering"""
    body = header('主题聚类与知识发现', 8)
    body += image_tag(60, 120, 500, 400, f'{IMG}\\p08_clustering.png')
    body += card(600, 120, 620, 180)
    body += card_title(600, 120, '社区检测算法', C_BLUE, 18)
    body += note(600, 175, '• 基于余弦相似度构建文档相似性矩阵', 15)
    body += note(600, 205, '• Louvain 算法进行社区发现与聚类', 15)
    body += note(600, 235, '• LLM 驱动生成主题标签与概要', 15)
    body += note(600, 265, '• 轮廓系数 · 模块度 · 稳定性评估三重质量门控', 15)

    body += card(600, 330, 620, 190)
    body += card_title(600, 330, '知识发现成果', C_PURPLE, 18)
    body += note(600, 385, '• 主题-文档映射关系：每个文档归属明确主题', 15)
    body += note(600, 415, '• 概念关联图谱：知识点之间的关联网络', 15)
    body += note(600, 445, '• Wiki 知识库目录：自动生成分类结构', 15)
    body += note(600, 475, '• 15+ 主题、60+ 概念条目、完整可追溯', 15)

    return footer(body)

def page09():
    """Wiki Vault"""
    body = header('Wiki Vault — 双向链接知识体系', 9)
    body += card(60, 120, 500, 480)
    body += card_title(60, 120, '目录结构', C_BLUE, 18)
    tree = [
        'wiki/',
        '├─ purpose.md',
        '│     ↳ 专家核心主张集合',
        '├─ schema.md',
        '│     ↳ 知识图谱元数据',
        '├─ topics/（15+ 主题）',
        '│     ↳ 波浪理论 / 房地产周期',
        '│       / 市场心理 / 板块轮动 …',
        '└─ concepts/（60+ 概念）',
        '      ↳ 微观聚焦 / 左侧风险',
        '        / 底部鬼故事 / 5成仓 …',
    ]
    for i, line in enumerate(tree):
        is_dir = line.startswith('├') or line.startswith('└') or line.startswith('│')
        is_indent = line.strip().startswith('↳')
        col = C_MUTED if is_indent else (C_TEXT if not is_dir else C_BLUE)
        body += txt(84, 185+i*30, line, 14, col, bold=(not line.startswith(' ') and '↳' not in line), family="Consolas,'Courier New',monospace")

    body += card(600, 120, 620, 480)
    body += card_title(600, 120, '设计理念与特性', C_PURPLE, 18)
    features = [
        ('类 Obsidian 的 Markdown 体系', '每个页面使用 Front Matter 标记元数据，兼容主流笔记软件'),
        ('概念-来源双向追溯', '每个概念条目均关联原文出处，支持反向查找'),
        ('可发现的知识网络', '概念间自动建立链接，形成可浏览的知识图谱'),
        ('AI-Agent 友好', '结构化格式便于 LLM 理解和引用，支持 RAG 检索'),
        ('可直接导入 Obsidian', '生成的文件格式与 Obsidian 完全兼容，无需额外转换'),
    ]
    for i, (title, desc) in enumerate(features):
        body += bullet(624, 195+i*80, title, 16, C_TEXT)
        body += note(624, 220+i*80, desc, 14, C_SUB, 24)

    return footer(body)

def page10():
    """LangGraph"""
    body = header('核心技术 — LangGraph 编排架构', 10)
    body += image_tag(140, 115, 1000, 320, f'{IMG}\\p10_langgraph.png')
    body += card(60, 460, 350, 150)
    body += card_title(60, 460, '编排引擎', C_BLUE, 17)
    body += note(60, 510, 'LangGraph StateGraph', 15)
    body += note(60, 540, '图状状态机，条件分支流转', 14, C_SUB)
    body += note(60, 568, '增量磁盘缓存，支持断点恢复', 14, C_SUB)

    body += card(450, 460, 770, 150)
    body += card_title(450, 460, '核心节点拓扑', C_PURPLE, 17)
    body += note(450, 510, '分析阶段（共享）→ 内容分析 · 表达分析 · 质量评估', 15)
    body += note(450, 540, '分流节点  →  Handbook 子图 / Skill 子图（条件分支）', 14, C_SUB)
    body += note(450, 568, '评审循环  →  未通过章节自动重写，选择性迭代', 14, C_SUB)

    return footer(body)

def page11():
    """Handbook subgraph"""
    body = header('Handbook 生成子图', 11)
    steps = [
        ('1', '章节规划', '生成完整目录结构', C_BLUE),
        ('2', '并行撰写', '各章节独立并行生成', '#60A5FA'),
        ('3', '交叉评审', 'LLM 自评，不合格重写', '#818CF8'),
        ('4', '连贯编辑', '全文风格统一', '#A78BFA'),
        ('5', '引用追踪', '所有观点关联原文', '#C4B5FD'),
    ]
    for i, (num, title, desc, color) in enumerate(steps):
        y = 120 + i*95
        body += card(60, y, 500, 80, C_CODE, color)
        body += f'<circle cx="100" cy="160" r="20" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2"/>'
        body += txt(100, 166, num, 14, color, True, 'middle')
        body += txt(135, 160, title, 17, C_TEXT, True)
        body += txt(135, 183, desc, 14, C_SUB)
        # arrow between steps
        if i < len(steps) - 1:
            body += f'<line x1="310" y1="{y+80}" x2="310" y2="{y+80+15}" stroke="{C_BORDER}" stroke-width="1.5" marker-end="url(#arrd)"/>'

    body += card(600, 120, 620, 490)
    body += card_title(600, 120, '技术亮点', C_PURPLE, 18)

    highlights = [
        ('💾 增量磁盘缓存', '每章独立写入缓存文件（.handbook_cache/）\n支持中断后断点续跑，避免重复消耗 token'),
        ('✏️ 选择性重写', '仅重写评审未通过的章节\n已通过的章节保留，大幅降低 LLM 调用成本'),
        ('🔗 引用完整性保证', '每个论断都关联到原文出处\n支持在 Handbook 中直接查看来源'),
        ('⏱️ 并行处理', '各章节生成互不依赖\n可充分利用多路并发提高效率'),
    ]
    for i, (title, desc) in enumerate(highlights):
        y = 190 + i*115
        body += txt(624, y, title, 17, C_TEXT, True)
        lines = desc.split('\n')
        for j, line in enumerate(lines):
            body += note(624, y+28+j*22, line, 14, C_SUB, 0)

    # Arrow marker
    body += '''<defs>
  <marker id="arrd" markerWidth="8" markerHeight="6" refX="4" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#64748B"/>
  </marker>
</defs>'''

    return footer(body)

def page12():
    """Skill subgraph"""
    body = header('Skill 生成子图 — Agent 能力封装', 12)
    steps = [
        ('🧠', '心智模型提取', '思维框架 · 决策启发式 · 价值观'),
        ('🎭', '表达编码', '语气风格 · 确信光谱 · 引用习惯'),
        ('📋', '智能体协议', '问题分类 → 研究维度 → 回答框架'),
        ('✅', '质量验证', '一致性检查 · 风格对齐 · 输出验证'),
    ]
    for i, (icon, title, desc) in enumerate(steps):
        x = 60 + i*305
        body += card(x, 120, 275, 210)
        body += txt(x+137, 175, icon, 28, C_TEXT, True, 'middle')
        body += txt(x+137, 215, title, 17, C_TEXT, True, 'middle')
        body += txt(x+137, 245, '', 14, C_SUB, anchor='middle')
        parts = desc.split(' · ')
        for j, part in enumerate(parts):
            body += note(x, 265+j*25, f'  {part}', 13, C_SUB, 0)
        if i < len(steps) - 1:
            body += f'<line x1="{x+275}" y1="225" x2="{x+289}" y2="225" stroke="{C_BORDER}" stroke-width="2" marker-end="url(#arrh)"/>'

    body += card(60, 370, 1160, 260)
    body += card_title(60, 370, 'Skill 文件结构', C_PURPLE, 18)
    items = [
        ('🎭 角色扮演规则', '定义 Agent 的身份、语气和应答风格'),
        ('🧠 核心心智模型清单', '专家的关键思维框架和决策模式'),
        ('⚡ 决策启发式', '快速决策的规则集合'),
        ('💡 价值观与反模式', '专家坚持的原则和避免的陷阱'),
        ('📝 应答协议', '问题接收 → 分析 → 回答的标准化流程'),
    ]
    for i, (title, desc) in enumerate(items):
        col = i % 2
        x = 84 + col * 560
        y = 430 + (i//2) * 65
        body += tspan(x, y, [(title, C_TEXT, True), (f'  —  {desc}', C_SUB, False)], 15)

    body += '''<defs>
  <marker id="arrh" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#64748B"/>
  </marker>
</defs>'''
    return footer(body)

def page13():
    """Quality control"""
    body = header('LLM 客户端与质量控制', 13)

    cols = [
        ('双档位设计', C_BLUE, [
            ('Fast 模式', '快速生成，适合批量处理', C_BLUE),
            ('Reasoning 模式', '深度推理，适合关键节点', C_PURPLE),
            ('自动切换', '根据任务复杂度自动选择', C_GREEN),
        ]),
        ('重试机制', C_PURPLE, [
            ('指数退避', '失败后逐步增加等待时间', C_BLUE),
            ('温度动态', '根据上下文调整随机性', '#60A5FA'),
            ('超时保护', '防止长时间卡住不动', C_ORANGE),
        ]),
        ('质量门控', C_GREEN, [
            ('内容相关性', '确保生成内容与主题相关', C_GREEN),
            ('引用完整性', '验证所有引用有源可溯', '#34D399'),
            ('风格一致性', '保持专家语气和风格统一', C_BLUE),
        ]),
    ]
    for i, (title, color, items) in enumerate(cols):
        x = 60 + i*400
        body += card(x, 120, 360, 480)
        body += card_title(x, 120, title, color, 19)
        for j, (item_title, item_desc, item_color) in enumerate(items):
            iy = 190 + j*120
            body += card(x+20, iy, 320, 100, C_CODE, item_color)
            body += txt(x+44, iy+30, item_title, 16, item_color, True)
            body += txt(x+44, iy+60, item_desc, 13, C_SUB)

    return footer(body)

def page14():
    """Results"""
    body = header('项目成果展示', 14)
    # Left: numbers
    body += card(60, 120, 550, 500)
    body += card_title(60, 120, '量化成果', C_BLUE, 20)
    metrics = [
        ('824', '篇', '专家内容抓取', '回答 + 文章'),
        ('3,100+', '', '知识块生成', '智能分块'),
        ('56', '页', 'Handbook', '7 大章节'),
        ('18', 'KB', 'Skill 文件', 'Agent 专用'),
        ('15+', '主题', 'Wiki 知识库', '60+ 概念条目'),
    ]
    for i, (num, unit, label, desc) in enumerate(metrics):
        y = 180 + i*75
        body += txt(84, y, num, 42, C_TEXT, True)
        if unit:
            body += txt(84+len(num)*25, y+10, unit, 18, C_SUB)
        body += txt(84+120, y+8, label, 17, C_TEXT, False)
        body += txt(84+120, y+32, desc, 13, C_MUTED)

    # Right: qualitative
    body += card(670, 120, 550, 500)
    body += card_title(670, 120, '质化成果', C_PURPLE, 20)
    quals = [
        ('完整的专家认知框架', '以黄彦臻投资框架为例，涵盖市场分析、\n风险管理、板块轮动等完整体系'),
        ('可复现的专家思维模式', '决策逻辑、分析路径、表达风格\n均可被 AI Agent 复现'),
        ('Claude Code 直接可用', '作为 memory 直接加载\n开箱即用，无需额外配置'),
        ('端到端管线验证通过', '从知乎爬取 → 知识蒸馏 → 输出\n全流程自动化，207 个测试用例通过'),
    ]
    for i, (title, desc) in enumerate(quals):
        y = 195 + i*105
        body += txt(694, y, f'✦ {title}', 17, C_TEXT, True)
        lines = desc.split('\n')
        for j, line in enumerate(lines):
            body += note(694, y+28+j*20, line, 14, C_SUB)

    return footer(body)

def page15():
    """Screenshot 1 — comparison without ED"""
    body = header('效果展示 — 普通 Claude Code 回答', 15)
    body += image_tag(90, 120, 1100, 480, f'{IMG}\\screenshot1.png')
    body += note(90, 620, '上图为未加载 Expert Digest 记忆时，Claude Code 对投资问题的标准回答。', 16, C_SUB)
    body += note(90, 648, '回答风格中立通用，缺乏特定领域的深度和专家个人色彩。', 14, C_MUTED, 20)
    return footer(body)

def page16():
    """Screenshot 2 — comparison with ED"""
    body = header('效果展示 — 加载 Expert Digest 后', 16)
    body += image_tag(90, 120, 1100, 480, f'{IMG}\\screenshot2.png')
    body += note(90, 620, '上图为加载 Handbook + Skill + Wiki 记忆后，Claude Code 对同一问题的回答。', 16, C_GREEN)
    body += note(90, 648, '对比可见：语气模仿 ● 观点引用 ● 思维一致 — 三大维度全面提升。', 14, C_MUTED, 20)
    return footer(body)

def page17():
    """Examples"""
    body = header('示例输出展示', 17)
    body += card(60, 120, 550, 510)
    body += card_title(60, 120, '📘 Handbook 示例章节', C_BLUE, 18)
    chapters = [
        '第一章  摒弃噪音：建立微观聚焦的投资思维',
        '第二章  市场语言：技术分析与周期识别',
        '第三章  决策框架：在不确定中寻找确定性',
        '第四章  行业分析：从宏观到微观的穿透',
        '第五章  风险控制：在贪婪与恐惧间平衡',
        '第六章  心态修炼：反人性的投资纪律',
        '第七章  实战应用：构建个人投资体系',
    ]
    for i, ch in enumerate(chapters):
        cy = 195 + i*50
        body += rect(84, cy, 500, 36, C_CODE, C_BORDER, 6)
        body += txt(104, cy+22, ch, 15, C_TEXT)

    body += card(670, 120, 550, 510)
    body += card_title(670, 120, '🛠️ Skill 应用效果', C_PURPLE, 18)
    effects = [
        ('🎭 语气模仿', 'Agent 能模仿专家特有的口语化分析风格，\n论述中穿插个人观点和态度表达。'),
        ('📚 观点引用', '回答中引用专家的具体观点、案例和\n历史判断（含原文出处）。'),
        ('🧠 思维一致', '采用专家的分析框架：先讲微观、\n再看宏观、最后回归决策。'),
        ('🔗 工具兼容', '可直接加载到 Claude Code / Cursor /\nVS Code Copilot 等主流 AI IDE。'),
    ]
    for i, (title, desc) in enumerate(effects):
        y = 195 + i*105
        body += txt(694, y, title, 17, C_TEXT, True)
        lines = desc.split('\n')
        for j, line in enumerate(lines):
            body += note(694, y+28+j*20, line, 14, C_SUB)

    return footer(body)

def page18():
    """Highlights"""
    body = header('技术创新与亮点', 18)
    highlights = [
        ('① 零依赖向量化', C_BLUE, '哈希词袋方案 (Hash Bag-of-Words)，零 API 成本。\n纯本地计算，结果可复现，无需调用任何外部 Embedding 服务。'),
        ('② 增量式生成', '#60A5FA', '磁盘缓存 + 断点续跑，每章独立写入。\n大幅降低重试成本，支持中断后从断点恢复。'),
        ('③ 浏览器态爬取', C_PURPLE, '绕开知乎反爬的创新方案。\n复用 Chrome 登录态，无需维护 Cookie 池和签名算法。'),
        ('④ 双轨蒸馏', '#A78BFA', 'Handbook（知识）+ Skill（能力）双通道输出。\n结构化知识手册 + Agent 技能描述，两种维度互补。'),
        ('⑤ 可追溯性', '#C4B5FD', '所有生成内容均可追溯到原文出处。\n引用完整性保证，每个论断都附带来源。'),
    ]
    for i, (title, color, desc) in enumerate(highlights):
        col = i % 3
        row = i // 3
        x = 60 + col*400
        y = 120 + row*220
        w = 370 if col < 2 else 370
        h = 190
        body += card(x, y, w, h)
        body += txt(x+24, y+35, title, 18, color, True)
        lines = desc.split('\n')
        for j, line in enumerate(lines):
            body += note(x, y+70+j*25, line, 14, C_SUB if j > 0 else C_TEXT, 24)

    return footer(body)

def page19():
    """Summary"""
    body = header('总结与展望', 19)
    body += card(60, 120, 550, 480)
    body += card_title(60, 120, '✅ 已完成', C_GREEN, 20)
    done_items = [
        ('端到端管线跑通', '从知乎爬取 → 知识蒸馏 → 输出的全链路验证通过'),
        ('知乎爬虫稳定可用', '浏览器上下文方案，853+7 篇内容稳定抓取'),
        ('Handbook / Skill 双输出', '56页学习手册 + 18KB 技能文件'),
        ('Wiki Vault 知识库', '15+ 主题，60+ 概念条目，双向链接'),
        ('207 个测试用例通过', '模块级测试覆盖核心功能'),
    ]
    for i, (title, desc) in enumerate(done_items):
        y = 190 + i*80
        body += txt(84, y, f'✅ {title}', 16, C_GREEN, True)
        body += note(84, y+28, desc, 14, C_SUB, 24)

    body += card(670, 120, 550, 480)
    body += card_title(670, 120, '🔮 未来方向', C_ORANGE, 20)
    future_items = [
        ('支持更多内容平台', '微信公众号、微博、Substack、Newsletter 等'),
        ('多专家知识融合', '交叉验证不同专家的观点异同'),
        ('实时增量更新', '定期抓取新内容，自动更新知识库'),
        ('效果量化评估', '建立 Agent 使用效果的评估指标体系'),
        ('MCP 服务化', '将知识库封装为 MCP 服务，供更多工具调用'),
    ]
    for i, (title, desc) in enumerate(future_items):
        y = 190 + i*80
        body += txt(694, y, f'🔮 {title}', 16, C_ORANGE, True)
        body += note(694, y+28, desc, 14, C_SUB, 24)

    return footer(body)

def page20():
    """Q&A"""
    gid, grad = _grad()
    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    {grad}
    <radialGradient id="bg" cx="50%" cy="30%" r="50%">
      <stop offset="0%" stop-color="{C_BLUE}" stop-opacity="0.1"/>
      <stop offset="100%" stop-color="{C_BLUE}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="{C_BG}"/>
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <text x="640" y="300" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="72" font-weight="bold" fill="{C_TEXT}" text-anchor="middle">Q&amp;A</text>
  <text x="640" y="380" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="28" fill="{C_SUB}" text-anchor="middle">感谢聆听</text>
  <rect x="440" y="420" width="400" height="2" rx="1" fill="url(#{gid})"/>
  <text x="640" y="500" font-family="'Segoe UI','Microsoft YaHei',sans-serif" font-size="18" fill="{C_MUTED}" text-anchor="middle">Expert Digest — 专家内容知识蒸馏引擎</text>
</svg>'''


# ─── Generate all ───
pages = [
    page01, page02, page03, page04, page05,
    page06, page07, page08, page09, page10,
    page11, page12, page13, page14, page15,
    page16, page17, page18, page19, page20,
]
names = [
    '01_cover', '02_background', '03_architecture', '04_crawler_intro', '05_crawler_arch',
    '06_data_output', '07_pipeline', '08_clustering', '09_wiki', '10_langgraph',
    '11_handbook', '12_skill', '13_quality', '14_results', '15_screenshot1',
    '16_screenshot2', '17_examples', '18_highlights', '19_summary', '20_qa',
]

for fn, pg in zip(names, pages):
    path = os.path.join(OUT, f'{fn}.svg')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(pg())
    print(f'  {path}')

print(f'\nDone! {len(pages)} SVGs generated.')
