from typing import Optional, List
from sqlalchemy.orm import Session

from backend.app.db.models import Video, VideoTopic, ChatSession, ChatMessage
from backend.app.models.schemas import VideoMetadata


def get_video_by_youtube_id(db: Session, youtube_id: str) -> Optional[Video]:
    return db.query(Video).filter(Video.youtube_id == youtube_id).first()


def create_or_update_video(db: Session, metadata: VideoMetadata, status: str = "READY") -> Video:
    video = get_video_by_youtube_id(db, metadata.video_id)
    if not video:
        video = Video(
            youtube_id=metadata.video_id,
            url=metadata.url,
            title=metadata.title,
            channel=metadata.channel,
            duration=metadata.duration,
            thumbnail=metadata.thumbnail_url,
            status=status,
        )
        db.add(video)
    else:
        video.title = metadata.title or video.title
        video.channel = metadata.channel or video.channel
        video.duration = metadata.duration or video.duration
        video.thumbnail = metadata.thumbnail_url or video.thumbnail
        video.status = status

    db.commit()
    db.refresh(video)
    return video


def save_video_topics(db: Session, video_id: int, topics_data: List[dict]):
    # Clear existing topics for this video
    db.query(VideoTopic).filter(VideoTopic.video_id == video_id).delete()
    for item in topics_data:
        topic = VideoTopic(
            video_id=video_id,
            topic=item["topic"],
            description=item.get("summary") or item.get("description", ""),
            start_time=item.get("start_time", 0.0),
            end_time=item.get("end_time", 0.0),
        )
        db.add(topic)
    db.commit()


def get_or_create_chat_session(db: Session, video_id: int, user_id: Optional[int] = None) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.video_id == video_id, ChatSession.user_id == user_id)
        .first()
    )
    if not session:
        session = ChatSession(video_id=video_id, user_id=user_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def save_chat_message(db: Session, session_id: int, role: str, content: str) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_chat_history(db: Session, session_id: int) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
