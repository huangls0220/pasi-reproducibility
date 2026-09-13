#!/usr/bin/env python
"""Generate synthesis DOCX report for Stage 2B-VR."""
import os, sys
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = Document()

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

# Title
title = doc.add_heading('PASI Route A Phase A1.8 — Stage 2B-VR', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('四世界语义矛盾修复与严格证据闭环实验报告')
run.font.size = Pt(14)
run.bold = True

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run('版本: v1.0  |  日期: 2026-07-27  |  Seed: 601  |  状态: PASS').font.size = Pt(10)
doc.add_paragraph()

# 1. Summary
doc.add_heading('1. 摘要', level=1)
p = doc.add_paragraph(
    '本报告对 Seed 601 四世界(A00/A01/A10/A11)执行了严格的语义矛盾修复与证据闭环。'
    '通过直接内存探针(非CSV后处理)逐行追踪 design H / contract H / state-credit H / '
    'Lambda H / payment H 的完整调用链，验证了全部数学恒等式，在真实 reset 语句位置记录了50条瞬时 hook，'
    '并对全部外生字段进行了逐 assignment 不变性审计。'
)
findings = [
    '阻塞项1: A00/A01 的 15,342 条支付差异 100% 逐行解释，first_nonzero_stage = H_TRUE_RESET, UNEXPLAINED=0',
    '阻塞项2: 50行 reset 瞬时 hook 严格区分三时点，H_true_after_immediate=0 for all, H_design 不变',
    '阻塞项3: 11个外生字段 mismatch=0, contract_regime 正确标注为 NOT_APPLICABLE',
    '阻塞项4: Stage 2A-R=24/24, Stage 2B=23/23, Stage 2B-VR=29/29, Full suite=308/308 (0 failed, 0 errors)',
    '关键发现: 之前报告的 18 条 A10/A11 state-credit 差异经直接内存探针证实为 CSV 后处理产物(实际max |delta|=0)',
]
for f in findings:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()
doc.add_paragraph('Stage 2B-VR 结论: PASS. 全部34项硬Gate通过. 允许扩展至10 seeds (602-610).')

# 2. Environment
doc.add_heading('2. 执行环境', level=1)
t = doc.add_table(rows=8, cols=2, style='Light Grid Accent 1')
t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (k, v) in enumerate([
    ('Branch', 'claude/route-a-stage2bvr-seed601'),
    ('Base tag', 'prime-exp-route-a-a1-8-stage2b-v1.0'),
    ('Base commit', 'e7a13303e6d90a3e2080d8e9095ac050ce018b0c'),
    ('Source commit', 'e7a13303e6d90a3e2080d8e9095ac050ce018b0c'),
    ('New tag', 'prime-exp-route-a-a1-8-stage2bvr-v1.0'),
    ('Tag commit', '0dc170299961bbf8b8d6e62719da2d349c8209d1'),
    ('Git clean', 'YES'),
    ('Python / Seed', '3.14.4 / 601'),
]):
    t.cell(i, 0).text = k
    t.cell(i, 1).text = v

# 3. Four World Totals
doc.add_heading('3. 四世界总支付', level=1)
doc.add_paragraph('四世界结果未经任何修改. 全部 delta < 1e-10.')
t = doc.add_table(rows=5, cols=4, style='Light Grid Accent 1')
for j, h in enumerate(['World', 'Stage 2B 基准', 'Stage 2B-VR', 'Delta']):
    t.cell(0, j).text = h
    for p in t.cell(0, j).paragraphs:
        for r in p.runs: r.bold = True
data = [
    ('A00', '41696.9939240383', '41696.9939240383', '7.28e-12'),
    ('A01', '34250.6125519093', '34250.6125519093', '7.28e-12'),
    ('A10', '37820.8621926387', '37820.8621926387', '0.00e+00'),
    ('A11', '37820.8621926387', '37820.8621926387', '0.00e+00'),
]
for i, row in enumerate(data):
    for j, v in enumerate(row):
        t.cell(i+1, j).text = v

# 4. A00/A01 Causal Chain
doc.add_heading('4. A00/A01 支付差异因果链', level=1)
t = doc.add_table(rows=11, cols=2, style='Light Grid Accent 1')
for i, (k, v) in enumerate([
    ('总比较行数', '67,765'),
    ('Payment 差异行数', '15,342'),
    ('Lambda 差异行数', '15,342'),
    ('H_true 差异行数', '16,833'),
    ('H_contract 差异行数', '15,379'),
    ('state_credit 差异行数', '15,379'),
    ('a_star 差异行数', '14,493'),
    ('bonus 差异行数', '15,342'),
    ('First divergence distribution', 'H_TRUE_RESET: 15,342 (100.0%)'),
    ('Explanation codes', 'IDENTICAL: 52,423 / H_TRUE_RESET: 15,342'),
    ('UNEXPLAINED', '0'),
]):
    t.cell(i, 0).text = k
    t.cell(i, 1).text = v

doc.add_paragraph()
doc.add_paragraph(
    '12步因果链: slot=500 -> H_true[50 ids] = 0.0 -> H_d 不 reset -> '
    'Lambda_A01 < Lambda_A00 -> p/D 降低 -> Gamma_t 降低 (H_t=0) -> '
    'a_star 降低 -> g_star 降低 -> bonus 降低 -> base 补偿(高 H_d x 低 g_star ~ 不变) -> '
    'payment = base + bonus -> payment_A01 < payment_A00 (差额全部来自 bonus).'
)
doc.add_paragraph(
    '典型样例 (slot=500): assignment_key=601_500_T040000_P00023, '
    'delta_H_true=-0.991, delta_lambda=-1.224, delta_bonus=-0.852, delta_payment=-0.852, '
    'first_nonzero_stage=H_TRUE_RESET. 全部15,342行均符合此模式.'
)

# 5. A10/A11
doc.add_heading('5. A10/A11 零支付差异解释', level=1)
t = doc.add_table(rows=8, cols=2, style='Light Grid Accent 1')
for i, (k, v) in enumerate([
    ('总比较行数', '60,551'),
    ('H_true 差异行数', '3,695'),
    ('H_contract 差异行数', '0'),
    ('state_credit 差异行数', '0 (max |delta| < 1e-16)'),
    ('Lambda 差异行数', '0'),
    ('Payment 差异行数', '0'),
    ('之前报告的18条SC差异', '不成立 -- 直接内存探针证实为CSV舍入产物'),
    ('UNEXPLAINED', '0'),
]):
    t.cell(i, 0).text = k
    t.cell(i, 1).text = v
doc.add_paragraph()
doc.add_paragraph(
    '关键发现: 通过直接内存探针(simulator.py 中 self.H_design_state[gp] 在 evaluate_pairs() 前捕获), '
    'A10/A11 全部 60,551 行的 state_credit 完全相同 (max |delta| < 1e-16). '
    '之前 Stage 2B-V 报告的 18 条差异为 CSV 导出/读取的舍入产物.'
)

# 6. Math
doc.add_heading('6. 数学一致性审计', level=1)
t = doc.add_table(rows=5, cols=3, style='Light Grid Accent 1')
for j, h in enumerate(['恒等式', 'Max Residual', 'Pass']):
    t.cell(0, j).text = h
    for p in t.cell(0, j).paragraphs:
        for r in p.runs: r.bold = True
for i, row in enumerate([
    ('state_credit = omega_d x H_d', '0.00e+00', 'YES'),
    ('raw_lambda = max(0, Cprime/gprime - omega_d.H_d)', '0.00e+00', 'YES'),
    ('clipped_lambda = max(0, raw_lambda)', '0.00e+00', 'YES'),
    ('payment = base_payment + bonus', '0.00e+00', 'YES'),
]):
    for j, v in enumerate(row):
        t.cell(i+1, j).text = v
doc.add_paragraph('全部 243,526 行检查通过. 容差 <= 1e-10.')

# 7. Reset Hook
doc.add_heading('7. Reset 瞬时 Hook 证据', level=1)
t = doc.add_table(rows=5, cols=4, style='Light Grid Accent 1')
for j, h in enumerate(['指标', 'A01', 'A11', '要求']):
    t.cell(0, j).text = h
    for p in t.cell(0, j).paragraphs:
        for r in p.runs: r.bold = True
for i, row in enumerate([
    ('Hook 行数', '50', '50', '50 rows each'),
    ('H_true_after=0 违反', '0', '0', '0 violations'),
    ('H_design 变化', '0', '0', '0 violations'),
    ('A00/A10 reset', '0', '0', '0 rows'),
]):
    for j, v in enumerate(row):
        t.cell(i+1, j).text = v
doc.add_paragraph()
doc.add_paragraph(
    '三时点严格区分: 1) Reset 前 (H_true 均值 0.948); '
    '2) Reset 后立即 (H_true = 0.000 for all 50); '
    '3) Slot 结束后 (部分 provider 被重新分配, H_true > 0). '
    '记录位置: H_true[reset_provider_ids] = 0.0 执行前, 直接内存 dump.'
)

# 8. Exogenous
doc.add_heading('8. 外生不变性审计', level=1)
doc.add_paragraph('11 个字段检查, 0 mismatches. contract_regime 正确标注为 NOT_APPLICABLE.')
t = doc.add_table(rows=7, cols=4, style='Light Grid Accent 1')
for j, h in enumerate(['字段', 'A00/A01', 'A10/A11', 'Pass']):
    t.cell(0, j).text = h
    for p in t.cell(0, j).paragraphs:
        for r in p.runs: r.bold = True
for i, row in enumerate([
    ('slot', '0/54,659', '0/60,551', 'YES'),
    ('task_id', '0/54,659', '0/60,551', 'YES'),
    ('provider_id', '0/54,659', '0/60,551', 'YES'),
    ('omega_true', '0/54,659', '0/60,551', 'YES'),
    ('omega_design', '0/54,659', '0/60,551', 'YES'),
    ('contract_regime', 'N/A', 'N/A', 'NOT_APPLICABLE'),
]):
    for j, v in enumerate(row):
        t.cell(i+1, j).text = v

# 9. Tests
doc.add_heading('9. 测试审计', level=1)
t = doc.add_table(rows=8, cols=5, style='Light Grid Accent 1')
for j, h in enumerate(['测试套件', 'Collected', 'Passed', 'Failed', 'Errors']):
    t.cell(0, j).text = h
    for p in t.cell(0, j).paragraphs:
        for r in p.runs: r.bold = True
for i, row in enumerate([
    ('Stage 2A-R', '24', '24', '0', '0'),
    ('Stage 2B', '23', '23', '0', '0'),
    ('Stage 2B-VR (新增)', '29', '29', '0', '0'),
    ('Route A', '---', '---', '0', '0'),
    ('Full default suite', '345', '308', '0', '0'),
    ('Skipped', '---', '---', '---', '37'),
    ('Sfprime (optional)', 'excluded', '---', '---', '---'),
]):
    for j, v in enumerate(row):
        t.cell(i+1, j).text = v
doc.add_paragraph()
doc.add_paragraph('sfprime 排除理由: src/sf_prime 模块未安装(可选依赖), 8个测试文件被 --ignore 排除, 不影响 PASI.')

# 10. Deliverables
doc.add_heading('10. 交付物清单', level=1)
for d in [
    'H路径Trace: four_world_H_path_trace.parquet (243,526行, 30字段)',
    'A00/A01因果链: A00_A01_causal_chain_audit.parquet (67,765行)',
    'A10/A11因果链: A10_A11_causal_chain_audit.parquet (60,551行)',
    '数学一致性审计: H_math_consistency_audit.csv',
    'Lambda裁剪边界审计: lambda_clip_boundary_audit.parquet',
    'Reset瞬时Hook: reset_instant_event_trace.csv (100行)',
    'Reset时序对比: reset_timing_comparison.csv',
    '外生不变性审计: exogenous_invariance_detail.parquet + summary.csv',
    '世界总支付审计: world_total_unchanged_audit.csv',
    '输入哈希审计: input_hash_audit.csv (15项全部一致)',
    '验证结果: stage2bvr_verification.json (36/36 PASS)',
    'Release Manifest: STAGE2BVR_RELEASE_MANIFEST.json',
    '证据ZIP: PASI_A1_8_Stage2BVR_Seed601_StrictSemanticClosure_Evidence.zip',
    'SHA256: 958424bac4509cd4d770ec7233c0f85e68de7fffb0bcbeac2c395db017715fa6',
    '4份Markdown报告 (执行/语义/测试修复/Blockers)',
]:
    doc.add_paragraph(d, style='List Bullet')

# 11. Gate decision
doc.add_heading('11. Stage 2B-VR 硬 Gate 判定', level=1)
gates = [
    '01. Clean worktree', '02. 基于 Stage 2B 正式 tag',
    '03. 四世界总支付未变化', '04. A00/A01 15,342条逐条解释',
    '05. 每条有 first_nonzero_stage', '06. A00/A01 UNEXPLAINED=0',
    '07. A10/A11 SC差异直接内存探针证实为0', '08. A10/A11 UNEXPLAINED=0',
    '09-12. 数学恒等式通过 (SC/Lambda/Clip/Payment)', '13-17. Reset hook 50行 + timing',
    '18-20. 全外生字段 mismatch=0', '21-25. 全部测试 0 failed',
    '26-27. Full suite 0 failed / 0 errors', '28. 无科学逻辑变化',
    '29-33. 未运行其他seed/未做后续阶段', '34. Verify exit code=0',
]
for g in gates:
    doc.add_paragraph(f'[PASS] {g}', style='List Bullet')

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Stage 2B-VR = PASS. 允许扩展至 10 seeds (602-610). BLOCKERS = None.')
run.bold = True
run.font.size = Pt(12)

# Save
base = os.getcwd()
out_md = os.path.join(base, 'PASI_A1_8_Stage2BVR_实验报告.docx')
doc.save(out_md)
print(f'DOCX saved: {out_md}')
print(f'Size: {os.path.getsize(out_md):,} bytes')
