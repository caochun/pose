"""Deterministic Weixin queries for the local pose monitor."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from contextlib import suppress, asynccontextmanager, AsyncExitStack
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/root/Develop/pose')
DB = ROOT / 'person_profiles.sqlite3'
STATUS = Path('/run/pose/status.json')
MEDIA = Path('/run/pose-query-media')
TZ = ZoneInfo('Asia/Shanghai')
POSE = {'sitting': '坐姿', 'standing': '站姿', 'unknown': '不确定'}
EVENT = {'new_profile': '建立档案', 'person_appeared': '检测到出现',
         'posture_changed': '姿态变化', 'person_unobserved': '持续未检出',
         'person_left': '旧版离开记录（可能误判）'}
log = logging.getLogger(__name__)
from .vision import analyze, latest_person, VisionUnavailable
from .labels import names, display, resolve, rename


def clock(ts):
    return datetime.fromtimestamp(ts, TZ).strftime('%m-%d %H:%M:%S')


def snapshot():
    try:
        data = json.loads(STATUS.read_text())
    except (OSError, ValueError):
        return {}, False
    return data, 0 <= time.time() - data.get('timestamp', 0) <= 5


def query(sql, args=()):
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=2) as db:
        return db.execute(sql, args).fetchall()


def _ensure_collection_tables(db):
    db.execute("""CREATE TABLE IF NOT EXISTS reid_collection (
        id INTEGER PRIMARY KEY CHECK(id=1), active INTEGER NOT NULL DEFAULT 0,
        profile_id INTEGER, track_id INTEGER, started_at REAL, updated_at REAL NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS reid_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
        track_id INTEGER, created_at REAL NOT NULL, pose TEXT NOT NULL,
        box TEXT NOT NULL, embedding TEXT NOT NULL)""")


def start_collection(raw_args=''):
    arg = raw_args.strip()
    if not arg:
        return '用法：/开始收集 老曹 或 /开始收集 P1。开始前画面中必须只有这个人。'
    try:
        pid = resolve(arg)
    except ValueError as exc:
        return str(exc)
    state, fresh = snapshot()
    visible = visible_tracks(state) if fresh else []
    if not fresh:
        return '实时检测状态已过期，请确认摄像头正常后重试。'
    if len(visible) != 1:
        return f'开始收集要求当前恰好检出1人，当前为{len(visible)}人；请清场后重试。'
    track = visible[0]
    if int(track.get('profile_id', -1)) != pid:
        return f'当前检出的是{display(track.get("profile_id"))}，不是{display(pid)}；请确认目标人物后重试。'
    now = time.time()
    with sqlite3.connect(DB, timeout=3) as db:
        _ensure_collection_tables(db)
        row = db.execute('SELECT active,profile_id FROM reid_collection WHERE id=1').fetchone()
        if row and row[0]:
            active_pid = int(row[1]) if row[1] is not None else None
            if active_pid == pid:
                count = db.execute('SELECT COUNT(*) FROM reid_samples WHERE profile_id=?', (pid,)).fetchone()[0]
                return f'正在收集{display(pid)}的ReID样本，已有{count}条；请继续变换角度、姿态和距离，完成后发送 /停止收集。'
            return f'当前正在收集{display(active_pid)}，请先发送 /停止收集。'
        db.execute('''INSERT INTO reid_collection(id,active,profile_id,track_id,started_at,updated_at)
                      VALUES(1,1,?,?,?,?)
                      ON CONFLICT(id) DO UPDATE SET active=1,profile_id=excluded.profile_id,
                      track_id=excluded.track_id,started_at=excluded.started_at,updated_at=excluded.updated_at''',
                   (pid, int(track['track_id']), now, now))
        db.commit()
        count = db.execute('SELECT COUNT(*) FROM reid_samples WHERE profile_id=?', (pid,)).fetchone()[0]
    suffix = f'，该人物已有{count}条历史样本' if count else ''
    return f'已开始收集{display(pid)}的ReID样本{suffix}。请依次变换正面、侧面、背面、坐姿、站姿和距离；完成后发送 /停止收集。'


def stop_collection(raw_args=''):
    if raw_args.strip():
        return '用法：/停止收集'
    now = time.time()
    with sqlite3.connect(DB, timeout=3) as db:
        _ensure_collection_tables(db)
        row = db.execute('SELECT active,profile_id,started_at FROM reid_collection WHERE id=1').fetchone()
        if not row or not row[0] or row[1] is None:
            return '当前没有进行中的ReID样本收集。'
        pid, started = int(row[1]), float(row[2] or now)
        db.execute('UPDATE reid_collection SET active=0,updated_at=? WHERE id=1', (now,))
        count = db.execute('SELECT COUNT(*) FROM reid_samples WHERE profile_id=?', (pid,)).fetchone()[0]
        session_count = db.execute('SELECT COUNT(*) FROM reid_samples WHERE profile_id=? AND created_at>=?',
                                    (pid, started)).fetchone()[0]
        db.commit()
    return (f'已停止收集{display(pid)}。本次持续{max(0, now-started):.0f}秒，新增{session_count}条样本，'
            f'累计{count}条；样本已保留，可继续用 /开始收集 {display(pid)} 追加。')


def people(raw_args=''):
    arg = raw_args.strip()
    pid = resolve(arg) if arg else None
    mapping=names()
    state, fresh = snapshot()
    fresh = fresh and 0 <= time.time() - state.get("inference_at", 0) <= 5
    lines = ['人物记录（北京时间）']
    if not fresh:
        lines.append('实时检测状态暂不可用或已超过5秒未更新，以下档案仅供回看。')
    else:
        tracks = [tr for tr in state.get('tracks', []) if pid is None or tr['profile_id'] == pid]
        visible = sum(tr['missed_results'] == 0 for tr in tracks)
        lines.append(f'当前检出：{visible}人；保留的暂时未检出轨迹：{len(tracks)-visible}条')
        for tr in tracks[:20]:
            missing = tr['missed_results'] > 0
            detail = (f"已{max(0, int(time.time()-tr['last_seen']))}秒未检出" if missing else POSE.get(tr['pose'], tr['pose']))
            lines.append(f"{display(tr['profile_id'],mapping)} / T{tr['track_id']}：{detail}")
    rows = query('SELECT id,first_seen,last_seen,last_pose FROM profiles ' +
                 ('WHERE id=? ' if pid is not None else '') + 'ORDER BY last_seen DESC LIMIT 10',
                 (pid,) if pid is not None else ())
    lines.append('档案（最近10个）：' if pid is None else f'{display(pid,mapping)}档案：')
    if not rows:
        lines.append('暂无记录。')
    for p, first, last, pose in rows:
        lines.append(f'{display(p,mapping)}：首次{clock(first)}，最近{clock(last)}，最后姿态{POSE.get(pose,pose)}')
    lines.append('P编号基于外观匹配，可能重建或误匹配；未检出不代表确实离开。')
    return '\n'.join(lines)


def today(raw_args=''):
    if raw_args.strip():
        return '用法：/今天'
    start = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    bounds = (start.timestamp(), end.timestamp())
    rows = query('SELECT profile_id,event,COUNT(*) FROM events WHERE created_at>=? AND created_at<? GROUP BY profile_id,event ORDER BY profile_id', bounds)
    profiles = query('SELECT id FROM profiles WHERE first_seen<? AND last_seen>=? ORDER BY id', (bounds[1], bounds[0]))
    mapping=names()
    lines = [f'今日记录：{start:%Y-%m-%d}（北京时间）', f'今天检出过的档案：{len(profiles)}个（非精确人数）']
    if profiles:
        lines.append('、'.join(display(p[0],mapping) for p in profiles[:30]) + ('…' if len(profiles) > 30 else ''))
    for pid, event, count in rows[:40]:
        lines.append(f'{display(pid,mapping)} {EVENT.get(event,event)}：{count}次')
    recent = query('SELECT profile_id,event,pose,created_at FROM events WHERE created_at>=? AND created_at<? ORDER BY created_at DESC LIMIT 10', bounds)
    lines.append('最近事件：')
    for pid, event, pose, timestamp in recent:
        lines.append(f'{clock(timestamp)} {display(pid,mapping)} {EVENT.get(event,event)}，{POSE.get(pose,pose)}')
    if not recent:
        lines.append('暂无事件。')
    lines.append('统计来自已有检测事件；姿态变化通知有5分钟冷却，不等于全部动作次数。目前未记录可靠的逐人姿态时长。')
    return '\n'.join(lines)


async def service(name, user=False):
    proc = await asyncio.create_subprocess_exec('systemctl', *(['--user'] if user else []), 'is-active', name,
                                               stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 3)
        return out.decode().strip() or 'unknown'
    finally:
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.communicate()


async def health(raw_args=''):
    if raw_args.strip():
        return '用法：/状态'
    states = await asyncio.gather(service('pose_monitor'), service('mediamtx'), service('hermes-gateway', True))
    data, fresh = snapshot()
    lines = ['设备状态（北京时间）', f'姿态服务：{states[0]}；MediaMTX：{states[1]}；Hermes：{states[2]}',
             f'系统负载（1/5/15分钟）：' + ' / '.join(f'{v:.2f}' for v in os.getloadavg())]
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split()
        if len(fields) > 1:
            mem[fields[0].rstrip(':')] = int(fields[1])
    lines.append(f"内存：{(mem['MemTotal']-mem['MemAvailable'])/1024:.0f} / {mem['MemTotal']/1024:.0f} MB")
    disk = os.statvfs(ROOT)
    lines.append(f'磁盘可用：{disk.f_bavail*disk.f_frsize/1024**3:.2f} GB')
    temps = []
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        with suppress(OSError, ValueError):
            temps.append(float(path.read_text())/1000)
    if temps:
        lines.append(f'最高温度：{max(temps):.1f}℃')
    if data:
        age = max(0, time.time()-data.get('timestamp', 0))
        inference_age = max(0, time.time()-data.get('inference_at', 0))
        visible = sum(tr['missed_results'] == 0 for tr in data.get('tracks', []))
        lines += [f"状态更新：{clock(data['timestamp'])}（{age:.0f}秒前）",
                  f"推流：{data.get('stream_fps',0):.1f}fps；推理：{data.get('inference_fps',0):.1f}fps",
                  f'最近有效推理：{inference_age:.0f}秒前；' + (f'检出{visible}人' if fresh and inference_age <= 5 else '检测结果已过期，不能判定当前人数')]
    else:
        lines.append('实时检测状态尚未生成。')
    return '\n'.join(lines)


def credentials():
    account = os.getenv('WEIXIN_ACCOUNT_ID', '').strip()
    token = os.getenv('WEIXIN_TOKEN', '').strip()
    base = os.getenv('WEIXIN_BASE_URL', '').strip()
    cdn = os.getenv('WEIXIN_CDN_BASE_URL', '').strip()
    if account and not token:
        home = Path(os.getenv('HERMES_HOME', '~/.hermes')).expanduser()
        with suppress(OSError, ValueError):
            saved = json.loads((home/'weixin'/'accounts'/f'{account}.json').read_text())
            token = saved.get('token', '')
            base = base or saved.get('base_url', '')
    extra = {'account_id': account}
    if base:
        extra['base_url'] = base
    if cdn:
        extra['cdn_base_url'] = cdn
    return extra, token or None


@asynccontextmanager
async def camera_file(video=False, seconds=5):
    MEDIA.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='query_', suffix='.mp4' if video else '.jpg', dir=MEDIA)
    os.close(fd)
    path = Path(filename)
    proc = None
    try:
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-rtsp_transport', 'tcp']
        if not video:
            cmd += ['-analyzeduration', '100000', '-probesize', '32768']
        cmd += ['-i', os.getenv('HERMES_RTSP_URL', 'rtsp://127.0.0.1:8554/cam')]
        cmd += (['-t', str(seconds), '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart'] if video
                else ['-frames:v', '1', '-q:v', '3'])
        cmd += ['-fs', str(20*1024*1024), filename]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, err = await asyncio.wait_for(proc.communicate(), seconds+20 if video else 20)
        if proc.returncode or not path.stat().st_size:
            log.warning('pose capture failed: %s', err.decode(errors='replace')[-300:])
            raise VisionUnavailable('摄像头采集失败，请检查 /状态 后重试。')
        yield path, time.time()
    finally:
        try:
            if proc and proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
                await proc.communicate()
        finally:
            path.unlink(missing_ok=True)


def visible_tracks(state):
    now = time.time()
    if not 0 <= now-state.get('timestamp',0) <= 5 or not 0 <= now-state.get('inference_at',0) <= 5:
        return []
    return [tr for tr in state.get('tracks',[]) if tr['missed_results']==0]


@asynccontextmanager
async def person_crop(path, state, track):
    from PIL import Image
    fd, filename = tempfile.mkstemp(prefix='person_',suffix='.jpg',dir=MEDIA)
    os.close(fd)
    crop = Path(filename)
    try:
        with Image.open(path) as image:
            width,height = image.size
            src_width,src_height = state.get('frame_size',(width,height))
            box = track.get('box')
            if not box or src_width <= 0 or src_height <= 0:
                raise VisionUnavailable('当前人物定位信息不可用，请稍后重试。')
            x1,y1,x2,y2 = box
            coords = (max(0,int(x1*width/src_width)-15),max(0,int(y1*height/src_height)-20),
                      min(width,int(x2*width/src_width)+15),min(height,int(y2*height/src_height)+15))
            if coords[2]-coords[0]<20 or coords[3]-coords[1]<20:
                raise VisionUnavailable('当前人物检测框过小，无法可靠分析。')
            image.crop(coords).convert('RGB').save(crop,quality=85)
        yield crop
    finally:
        crop.unlink(missing_ok=True)


class CameraCommands:
    def __init__(self):
        self.busy = False

    def recipient(self):
        from gateway.session_context import get_session_env
        if get_session_env('HERMES_SESSION_PLATFORM', '').lower() != 'weixin':
            raise VisionUnavailable('请在微信通道使用此命令。')
        chat = get_session_env('HERMES_SESSION_CHAT_ID', '')
        if not chat:
            raise VisionUnavailable('无法确定微信接收会话，请重新发送命令。')
        return chat

    async def send(self, chat, message, path):
        from gateway.platforms.weixin import send_weixin_direct
        extra,token = credentials()
        result = await asyncio.wait_for(send_weixin_direct(extra=extra,token=token,chat_id=chat,message=message,
            media_files=[(str(path),False)]),60)
        if not result.get('success'):
            log.warning('pose attachment delivery failed')
            raise VisionUnavailable('微信附件发送失败，请稍后重试。')
        log.info('pose attachment delivered (%s bytes)',path.stat().st_size)

    async def capture(self, video, raw_args=''):
        arg = raw_args.strip()
        seconds = 5
        if video:
            if arg and (not arg.isdigit() or not 3 <= int(arg) <= 15):
                return '用法：/视频，默认5秒；或 /视频 10（允许3–15秒）'
            seconds = int(arg) if arg else 5
        elif arg:
            return '用法：/截图'
        if self.busy:
            return '正在处理另一张截图、视频或图片解读，请稍后再试。'
        self.busy = True
        kind = '视频' if video else '截图'
        try:
            chat = self.recipient()
            async with camera_file(video,seconds) as (path,stamp):
                await self.send(chat,f'摄像头{kind}（{clock(stamp)}，北京时间）',path)
            return None
        except VisionUnavailable as exc:
            return str(exc)
        except asyncio.TimeoutError:
            return f'{kind}采集或发送超时，请稍后重试。'
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('pose capture/delivery failed')
            return f'{kind}处理失败，请检查 /状态 后重试。'
        finally:
            self.busy = False

    async def observe(self, mode, args='', base=''):
        if self.busy:
            return (base+'\n' if base else '')+'正在处理另一张截图、视频或图片解读，请稍后再试。'
        pid = None
        seconds = 5
        if mode=='person':
            if args.strip():
                pid=int(args.strip().lstrip('Pp'))
            state,_=snapshot()
            tracks=[tr for tr in visible_tracks(state) if pid is None or tr['profile_id']==pid]
            if not tracks:
                return base+'\n当前没有可用于实时图片解读的对应人物检测。'
        elif mode=='change':
            if args.strip() and (not args.strip().isdigit() or not 3 <= int(args.strip()) <= 15):
                return '用法：/变化，默认等待5秒；或 /变化 10（允许3–15秒）'
            seconds=int(args.strip()) if args.strip() else 5
        elif len(args)>200:
            return '问题请控制在200字以内，例如：/看看 桌面上有什么？'
        self.busy=True
        observation=''
        try:
            chat=self.recipient()
            async with AsyncExitStack() as stack:
                first,stamp=await stack.enter_async_context(camera_file())
                state,_=snapshot()
                tracks=visible_tracks(state)
                metadata={'captured_at':stamp,'tracks':tracks,'frame_size':state.get('frame_size')}
                pids=[]
                paths=[first]
                send_path=first
                source='look'
                if mode=='change':
                    await asyncio.sleep(seconds)
                    second,second_stamp=await stack.enter_async_context(camera_file())
                    paths.append(second)
                    send_path=second
                    metadata={'before_at':stamp,'after_at':second_stamp}
                    prompt=(f'比较按时间顺序提供的两张摄像头截图，间隔约{second_stamp-stamp:.0f}秒。'
                            '描述实际可见的人物位置、动作和物品变化；未能确认对应的情况请说明，不凭标签断言身份或离开。')
                    source='change'
                    heading=f'画面变化（{clock(stamp)} → {clock(second_stamp)}，北京时间）'
                elif mode=='person' and pid is not None:
                    selected=[tr for tr in tracks if tr['profile_id']==pid]
                    if len(selected)!=1:
                        return base+'\n采集时该人物未被明确检出，未进行当前动作解读。'
                    tr=selected[0]
                    crop=await stack.enter_async_context(person_crop(first,state,tr))
                    paths=[crop]
                    # Attach only the target crop, so the recipient can assess the description.
                    send_path=crop
                    prompt=(f'这是本地轨迹P{pid}/T{tr["track_id"]}附近的裁剪画面。'
                            '描述主要人物可见的动作、朝向和互动的物品。不确认身份，检测框可能不准确，若人物不清楚请说明。')
                    pids=[pid]
                    metadata={'captured_at':stamp,'track':tr}
                    source='person'
                    heading=f'{display(pid)} 当前画面观察（{clock(stamp)}，北京时间）'
                elif mode=='person':
                    prompt=('描述当前各人物可见的动作和大致位置。只有能看清且明确对应的P/T标签才引用，'
                            '否则按左侧、右侧等位置描述，不猜人物编号。检测辅助信息：'+json.dumps(metadata,ensure_ascii=False))
                    heading=f'当前人物画面观察（{clock(stamp)}，北京时间）'
                else:
                    prompt='描述当前摄像头画面中的人物动作、物品和场景。'
                    if args.strip():
                        prompt+='重点回答：'+args.strip()
                    heading=f'画面观察（{clock(stamp)}，北京时间）'
                desc=await analyze(paths,prompt,source=source,pids=pids,metadata=metadata)
                observation=heading+'\n'+desc
                await self.send(chat,(base+'\n\n' if base else '')+observation,send_path)
                return None
        except VisionUnavailable as exc:
            return (base+'\n\n' if base else '')+(observation+'\n' if observation else '')+str(exc)
        except asyncio.TimeoutError:
            return (base+'\n\n' if base else '')+(observation+'\n' if observation else '')+'采集、解读或微信发送超时，请稍后重试。'
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('pose visual query failed')
            return (base+'\n\n' if base else '')+(observation+'\n' if observation else '')+'图片解读暂不可用，请稍后重试。'
        finally:
            self.busy=False

    async def image(self,args=''):
        return await self.capture(False,args)

    async def video(self,args=''):
        return await self.capture(True,args)

    async def look(self,args=''):
        return await self.observe('look',args)

    async def change(self,args=''):
        return await self.observe('change',args)

def register(ctx):
    camera = CameraCommands()

    async def guarded(fn, args):
        try:
            return await asyncio.to_thread(fn, args)
        except (OSError, sqlite3.Error, ValueError, KeyError):
            log.exception('pose query failed')
            return '人物记录读取失败，请检查 /状态 后重试。'

    async def handle_people(args=''):
        arg=args.strip()
        try:
            pid=await asyncio.to_thread(resolve,arg) if arg else None
        except ValueError as exc:
            return str(exc)
        normalized=f'P{pid}' if pid is not None else ''
        base=await guarded(people,normalized)
        if '读取失败' in base:
            return base
        if pid is not None:
            try:
                last=await asyncio.to_thread(latest_person,pid)
                if last:
                    base+='\n最近图片观察（历史，'+clock(last[0])+'）：'+last[1]
            except (OSError,sqlite3.Error):
                log.warning('visual history unavailable')
        return await camera.observe('person',normalized,base)

    async def handle_name(args=''):
        return await guarded(rename,args)

    async def handle_today(args=''):
        return await guarded(today, args)

    async def handle_start_collection(args=''):
        return await guarded(start_collection, args)

    async def handle_stop_collection(args=''):
        return await guarded(stop_collection, args)

    options = {'busy_policy': 'dispatch'} if 'busy_policy' in inspect.signature(ctx.register_command).parameters else {}
    for name, handler, description, hint in [
        ('人物', handle_people, '查看实时人物和最近档案', '[P编号或名称]'),
        ('命名', handle_name, '给人物档案命名或清除名称', '[P编号 名称]'),
        ('开始收集', handle_start_collection, '为指定人物开始收集ReID样本', '[名称或P编号]'),
        ('停止收集', handle_stop_collection, '停止当前ReID样本收集', ''),
        ('今天', handle_today, '查看北京时间今日人物事件', ''),
        ('截图', camera.image, '发送当前摄像头标注截图', ''),
        ('状态', health, '查看设备负载和服务状态', ''),
        ('视频', camera.video, '发送摄像头短视频，默认5秒', '[3–15秒]'),
        ('看看', camera.look, '解读当前摄像头画面并发送图片', '[问题]'),
        ('变化', camera.change, '对比间隔5秒的两张摄像头截图', '[3–15秒]'),
    ]:
        ctx.register_command(name, handler=handler, description=description, args_hint=hint, **options)
