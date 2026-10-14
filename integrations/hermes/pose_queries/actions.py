"""Read local MotionBERT candidates without invoking an LLM or sending messages."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

STATUS = Path('/run/pose/actions.json')
LABELS = (
    '喝水', '吃东西', '刷牙', '梳头', '物品掉落', '拾起物品', '投掷', '坐下', '站起', '拍手',
    '阅读', '书写', '撕纸', '穿外套', '脱外套', '穿鞋', '脱鞋', '戴眼镜', '摘眼镜', '戴帽子',
    '摘帽子', '欢呼', '挥手', '踢物品', '伸手进口袋', '单脚跳', '跳起', '打电话', '使用手机或平板', '键盘打字',
    '指向物品', '自拍', '看手表', '搓手', '点头或鞠躬', '摇头', '擦脸', '敬礼', '双手合十', '双手交叉示意停止',
    '打喷嚏或咳嗽', '踉跄', '跌倒', '摸头', '摸胸口', '摸背', '摸脖子', '呕吐动作', '扇风', '挥拳或拍打他人',
    '踢他人', '推他人', '拍他人后背', '指向他人', '拥抱', '递东西给他人', '碰他人口袋', '握手', '相向走近', '相互走远',
)


def action_lines(path=None, now=None, profile_display=None):
    now = time.time() if now is None else now
    path = STATUS if path is None else path
    try:
        data = json.loads(path.read_text())
        if not 0 <= now - data.get('timestamp', 0) <= 5:
            return ['动作识别状态已过期。']
        model_name = data.get('model', '动作模型')
        backend = data.get('backend', 'CPU')
        lines = [f'动作候选（{model_name}，{backend}，实验）：']
        if not data.get('loaded'):
            return lines + ['模型加载中。']
        if not data.get('source_fresh'):
            return lines + ['摄像头关键点尚未更新，等待姿态检测恢复。']
        if not data.get('people'):
            return lines + ['当前未检出人物，动作推理暂停。']
        for person in data['people']:
            pid = person['profile_id']
            name = profile_display(pid) if profile_display else f'P{pid}'
            prefix = f'{name}（T{person["track_id"]}）'
            result = person.get('result')
            if not result or not 0 <= now - result['captured_at'] <= 30:
                lines.append(prefix + '：等待足够的连续关键点。')
                continue
            candidates = []
            for item in result.get('candidate_actions', [])[:3]:
                index = item['id']
                label = LABELS[index] if 0 <= index < len(LABELS) else f'类别{index}'
                candidates.append(f'{label} {item["score"]:.0%}')
            age = now - result['captured_at']
            captured = datetime.fromtimestamp(result['captured_at'], ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
            lines.append(prefix + '：' + ' / '.join(candidates))
            lines.append(f'片段结束于 {captured}（{age:.0f}秒前）；计算耗时 {result["inference_ms"]/1000:.1f}秒')
        if data.get('last_error'):
            lines.append('最近一次动作推理失败，服务将继续重试。')
        lines.append(f'约每{data.get("interval_seconds", 15):g}秒更新，分数表示模型匹配程度。')
        return lines
    except (OSError, ValueError, KeyError, TypeError):
        return ['动作识别结果暂不可用。']
