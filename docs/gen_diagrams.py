"""Generate architecture diagrams and clustering visualization for PPT."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
import os

# Find Windows Chinese font
_FONT = None
for fp in [
    'C:\\Windows\\Fonts\\msyh.ttc',
    'C:\\Windows\\Fonts\\simhei.ttf',
    'C:\\Windows\\Fonts\\simsun.ttc',
]:
    if os.path.exists(fp):
        _FONT = fm.FontProperties(fname=fp)
        break
if _FONT is None:
    raise RuntimeError("No Chinese font found!")

def cf(size=9):
    return fm.FontProperties(fname=_FONT.get_file(), size=size)

OUT_DIR = r"D:\Project\ppt-master-2.6.0\projects\expert_digest_ppt169_20260514\images"

BG = '#0F172A'; CARD = '#1E293B'; BORDER = '#334155'
BLUE = '#3B82F6'; PURPLE = '#8B5CF6'; GREEN = '#10B981'
ORANGE = '#F59E0B'; TEXT = '#E2E8F0'; SUB = '#94A3B8'

def setup_dark(fig): fig.patch.set_facecolor(BG)

def save(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Saved: {path}")

# ═══════════════════════════════════════════════
# P03 — Architecture
# ═══════════════════════════════════════════════
def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    setup_dark(fig)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis('off')

    def box(x, y, w, h, color, label, sub=None):
        ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                     facecolor=color, edgecolor=BORDER, lw=1.5, alpha=0.9))
        ax.text(x+w/2, y+h/2+0.08, label, ha='center', va='center', fontproperties=cf(9), color=TEXT)
        if sub:
            ax.text(x+w/2, y+h/2-0.35, sub, ha='center', va='center', fontproperties=cf(7), color=SUB)

    # Sources
    box(0.5, 2.5, 2.2, 0.7, CARD, '知乎 (Zhihu)', '853 回答 + 7 文章')
    box(3.0, 2.5, 2.2, 0.7, CARD, '其他来源', 'JSONL / Markdown')

    # Processing
    box(0.5, 1.3, 4.7, 0.8, CARD, '存储处理层 · SQLite + Bag-of-Words', '导入 → 清洗 → 分块 → 向量化 → 聚类分析')
    ax.add_patch(mpatches.FancyBboxPatch((2.8, 1.45), 2.4, 0.5, boxstyle="round,pad=0.08",
                 facecolor='none', edgecolor=PURPLE, lw=2, ls='--'))
    ax.text(4.0, 1.2, '✦ 零依赖向量化', ha='center', fontproperties=cf(6), color=PURPLE)

    # LangGraph
    box(0.5, 0.3, 4.7, 0.7, CARD, 'LangGraph 知识蒸馏管线', '分析 → Handbook 子图 / Skill 子图 → 评审 → 输出')
    ax.annotate('', xy=(1.6, 1.0), xytext=(1.6, 1.3), arrowprops=dict(arrowstyle='->', color=SUB, lw=1.5))

    # Outputs
    box(6.2, 2.2, 1.5, 0.7, PURPLE, 'Wiki Vault', '15+ 主题, 60+ 概念')
    box(6.2, 1.1, 1.5, 0.7, BLUE, 'Handbook', '56 页, 7 章节')
    box(6.2, 0.2, 1.5, 0.7, GREEN, 'Skill.md', '18KB, Agent 就绪')

    for xy1, xy2 in [( (5.2,2.85), (6.0,2.55) ),
                      ( (5.2,2.55), (6.0,1.45) ),
                      ( (5.2,1.45), (6.0,0.55) )]:
        ax.annotate('', xy=xy2, xytext=xy1, arrowprops=dict(arrowstyle='->', color=SUB, lw=1.2, ls='dotted'))

    save('p03_architecture.png')

# ═══════════════════════════════════════════════
# P05 — Crawler
# ═══════════════════════════════════════════════
def draw_crawler():
    fig, ax = plt.subplots(figsize=(9, 2.8))
    setup_dark(fig); ax.set_xlim(0, 9); ax.set_ylim(0, 2.8); ax.axis('off')

    def box(x, y, w, h, color, label, items):
        ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                     facecolor=color, edgecolor=BORDER, lw=1.5, alpha=0.9))
        ax.text(x+w/2, y+h-0.3, label, ha='center', fontproperties=cf(8), color=TEXT)
        for i, item in enumerate(items):
            ax.text(x+w/2, y+h-0.6-i*0.25, item, ha='center', fontproperties=cf(6.5), color=SUB)

    def arr(x1,y1,x2,y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=SUB, lw=1.5))

    box(0.2, 1.8, 1.2, 0.7, PURPLE, '浏览器上下文', ['Chrome 登录态', 'opencli bridge'])
    box(1.8, 1.6, 1.3, 1.0, BLUE, 'fetcher', ['分页抓取', '速率限制', '风控检测'])
    box(3.5, 1.6, 1.3, 1.0, BLUE, 'normalizer', ['统一 schema', '字段转换', '数据清洗'])
    box(5.2, 1.6, 1.3, 1.0, PURPLE, 'transform', ['HTML→Markdown', '文本提取', '图片URL'])
    box(6.9, 1.6, 1.3, 1.0, PURPLE, 'writer', ['JSONL写入', '原子操作', '防损坏'])
    box(6.9, 0.3, 1.3, 0.6, CARD, 'checkpoint', ['断点续爬', 'ID去重'])

    arr(1.4,2.15,1.8,2.1); arr(3.1,2.1,3.5,2.1); arr(4.8,2.1,5.2,2.1); arr(6.5,2.1,6.9,2.1)
    arr(7.55,1.6,7.55,0.9)

    ax.text(0.5, 0.1, '反爬策略：低速率(2.5s) + 随机抖动 · 遇 401/403/429 立即停止 · 最小接口集',
            fontproperties=cf(6), color=ORANGE)

    save('p05_crawler.png')

# ═══════════════════════════════════════════════
# P08 — Clustering
# ═══════════════════════════════════════════════
def draw_clustering():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(5, 4))
    setup_dark(fig); ax.set_facecolor('#0B1120')

    clusters = {
        '市场分析': (1, 1, '#3B82F6', 0.35),
        '投资理念': (2, 3, '#8B5CF6', 0.30),
        '板块轮动': (4, 2, '#10B981', 0.30),
        '风险控制': (3, 4.5, '#F59E0B', 0.25),
        '经济周期': (5.5, 3.5, '#EF4444', 0.28),
    }
    n = 120
    for label, (cx, cy, color, sp) in clusters.items():
        ax.scatter(np.random.normal(cx, sp, n//5), np.random.normal(cy, sp, n//5),
                   c=color, alpha=0.4, s=15, label=label)

    ax.set_xlim(0, 6.5); ax.set_ylim(0, 5.5)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=BORDER, prop=cf(8))
    ax.set_title('主题聚类 (t-SNE 降维示意)', fontproperties=cf(12), color=TEXT, pad=10)
    ax.tick_params(colors=SUB, labelsize=6)
    for spine in ax.spines.values(): spine.set_color(BORDER)

    save('p08_clustering.png')

# ═══════════════════════════════════════════════
# P10 — LangGraph
# ═══════════════════════════════════════════════
def draw_langgraph():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    setup_dark(fig); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis('off')

    def node(x, y, w, h, color, label, items):
        ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                     facecolor=color, edgecolor=BORDER, lw=1.5, alpha=0.9))
        ax.text(x+w/2, y+h-0.25, label, ha='center', fontproperties=cf(8), color=TEXT)
        for i, item in enumerate(items):
            ax.text(x+w/2, y+h-0.55-i*0.22, item, ha='center', fontproperties=cf(6.5), color=SUB)

    def arr(x1,y1,x2,y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=SUB, lw=2))

    node(0.2, 0.6, 1.2, 0.8, CARD, '数据输入', ['SQLite 知识库', '嵌入向量'])
    node(1.8, 0.6, 1.8, 0.8, BLUE, '分析阶段（共享）', ['内容分析 · 表达分析', '质量评估 · 主题聚类'])
    arr(1.4,1.0,1.8,1.0)

    node(4.7, 2.0, 1.0, 0.5, PURPLE, '分流节点', ['条件分支'])
    arr(3.6,1.0,4.7,2.25)

    node(6.0, 2.0, 3.5, 1.0, PURPLE, 'Handbook 子图', ['章节规划→并行撰写→交叉评审→连贯编辑→引用追踪'])
    arr(5.7,2.25,6.0,2.5)

    node(6.0, 0.3, 3.5, 1.0, GREEN, 'Skill 子图', ['心智模型提取→表达编码→智能体协议→质量验证'])
    arr(3.6,1.0,4.7,1.0); arr(5.7,0.8,6.0,0.8)

    node(9.8, 1.15, 0.3, 0.9, ORANGE, '输出', ['Handbook', 'Skill'])

    save('p10_langgraph.png')

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating diagrams...")
    draw_architecture(); draw_crawler(); draw_clustering(); draw_langgraph()
    print("All diagrams generated!")
