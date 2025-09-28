# whatsapp.py
import os
import requests
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import PlainTextResponse

from chatbot import get_llm_response
from db import get_session
from models import User
from sqlmodel import Session, select

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "verify-token-change-me")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
GRAPH_BASE = os.getenv("WHATSAPP_GRAPH_BASE", "https://graph.facebook.com/v19.0")


def _find_user_by_phone(session: Session, phone: str) -> Optional[User]:
    # Normalize: keep last 10 digits for matching if needed
    digits = ''.join([c for c in phone if c.isdigit()])
    user = session.exec(select(User).where(User.phone == digits)).first()
    if user:
        return user
    # Try with country code variations (very simple heuristic)
    if len(digits) > 10:
        tail = digits[-10:]
        user = session.exec(select(User).where(User.phone == tail)).first()
        if user:
            return user
    return None


@router.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(content=challenge or "", status_code=200)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook")
async def receive_message(payload: dict, session: Session = Depends(get_session)):
    # Minimal handler for text messages
    try:
        entry = (payload.get("entry") or [])[0]
        change = (entry.get("changes") or [])[0]
        value = change.get("value", {})
        messages = value.get("messages") or []
        if not messages:
            return {"status": "ignored"}
        message = messages[0]
        from_info = message.get("from")  # phone number string
        text = (message.get("text") or {}).get("body", "").strip()
        if not text:
            return {"status": "ignored"}

        # Resolve user by phone, if exists
        user: Optional[User] = _find_user_by_phone(session, from_info) if from_info else None
        user_id = user.id if user else None

        # Generate reply
        answer = get_llm_response(text, user_id=user_id)

        # Send reply via Graph API
        if not (WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID):
            # No credentials configured; just log and return
            return {"status": "ok", "note": "No WA credentials set", "answer": answer}

        url = f"{GRAPH_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "to": from_info,
            "type": "text",
            "text": {"body": answer[:4096]},
        }
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        if not resp.ok:
            return {"status": "send_failed", "code": resp.status_code, "detail": resp.text}
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
