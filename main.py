import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import pandas as pd
import numpy as np

import streamlit as st
import os
import base64

import streamlit as st
import base64
import os

audio_file_path = r"C:\Users\flux304\Downloads\relaxing-piano-310597.mp3"

if os.path.exists(audio_file_path):
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

    # 버튼을 클릭했을 때 오디오가 재생되도록 합니다.
    if st.button("🎵 배경 음악 재생"):
        st.markdown(
            f"""
            <audio autoplay loop controls style="display: none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )
        st.info("🎵 배경 음악이 재생 중입니다.")

else:
    st.error(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file_path}")

    import streamlit as st
import base64

# 간단한 배경 설정 함수
def set_jpg_background(image_path="background.jpg"):
    """JPG 파일을 50% 투명도로 배경 설정"""
    
    with open(image_path, "rb") as f:
        img_str = base64.b64encode(f.read()).decode()
    
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255,255,255,0.5), rgba(255,255,255,0.5)), 
                   url(data:image/jpeg;base64,{img_str});
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# 사용법
set_jpg_background(r"C:\Users\flux304\Desktop\asd\ba.jpg")  # JPG 파일 경로 입력


# 환경 변수 로드
load_dotenv('eed.env')

# API 키 확인
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("❌ OPENAI_API_KEY를 찾을 수 없습니다. eed.env 파일을 확인해주세요.")
    st.stop()

# 벡터 스토어 저장 경로 정의
FAISS_INDEX_DIR = "faiss_index"

# 페이지 설정
st.set_page_config(
    page_title="심리와 MBTI 전문 AI 상담사 - RAG 기반",
    page_icon="",
    layout="wide"
)

# MBTI 유형별 커뮤니케이션 스타일 정의 (8가지 전체)
MBTI_STYLES = {
    "INTJ": {
        "style": "논리적이고 체계적인 설명을 선호합니다. 핵심 포인트를 명확히 제시하고 단계별로 설명해드리겠습니다.",
        "tone": "전문적이고 분석적인 톤",
        "approach": "데이터와 논리에 기반한 해결책 제시",
        "keywords": ["논리적", "체계적", "분석", "효율", "계획"]
    },
    "ENFP": {
        "style": "창의적이고 열정적인 접근을 선호합니다. 다양한 가능성을 함께 탐구해보겠습니다.",
        "tone": "친근하고 격려하는 톤",
        "approach": "브레인스토밍과 감정적 지지 제공",
        "keywords": ["창의적", "아이디어", "가능성", "열정", "영감"]
    },
    "ISTJ": {
        "style": "명확하고 직설적인 소통을 선호합니다. 구체적인 절차와 방법을 제시해드리겠습니다.",
        "tone": "정중하고 체계적인 톤",
        "approach": "단계별 가이드와 실용적 해결책",
        "keywords": ["체계적", "절차", "규칙", "단계", "정확"]
    },
    "ESFJ": {
        "style": "따뜻하고 배려 깊은 소통을 선호합니다. 귀하의 감정을 충분히 이해하고 지원해드리겠습니다.",
        "tone": "공감적이고 지지적인 톤",
        "approach": "감정적 케어와 인간적 접근",
        "keywords": ["감정", "사람", "배려", "도움", "관계"]
    },
    "ENTP": {
        "style": "혁신적이고 유연한 접근을 선호합니다. 다양한 관점에서 문제를 바라보며 창의적 해결책을 모색하겠습니다.",
        "tone": "역동적이고 도전적인 톤",
        "approach": "아이디어 발산과 새로운 시각 제공",
        "keywords": ["혁신", "도전", "유연", "변화", "실험"]
    },
    "ISFP": {
        "style": "개인적이고 진정성 있는 소통을 선호합니다. 귀하의 개별적 상황과 가치를 존중하며 접근하겠습니다.",
        "tone": "따뜻하고 개인적인 톤",
        "approach": "개인 맞춤형 케어와 가치 중심 해결",
        "keywords": ["개인적", "진정성", "가치", "예술", "감성"]
    },
    "ESTJ": {
        "style": "효율적이고 실용적인 소통을 선호합니다. 즉시 실행 가능한 명확한 해결책을 제시해드리겠습니다.",
        "tone": "직접적이고 확실한 톤",
        "approach": "즉각적 행동과 결과 중심 해결",
        "keywords": ["효율", "실용", "빠른", "결과", "성취", "목표", "리더십", "실행"]
    },
    "INFJ": {
        "style": "깊이 있고 의미 있는 소통을 선호합니다. 근본적 원인을 파악하여 장기적 관점에서 해결해드리겠습니다.",
        "tone": "신중하고 통찰력 있는 톤",
        "approach": "본질적 문제 해결과 미래 지향적 조언",
        "keywords": ["깊이", "의미", "통찰", "미래", "이해"]
    }
}

# 심리학 전문 지식 데이터베이스 (RAG용 샘플 데이터)
PSYCHOLOGY_KNOWLEDGE = """
MBTI 심리학 전문 지식베이스:

INTJ (건축가) 유형 특성:
- 독립적이고 창조적인 사고를 가진 완벽주의자
- 장기적 계획과 전략적 사고에 뛰어남  
- 스트레스 상황에서는 혼자만의 시간이 필요
- 불만 해결 시 논리적 근거와 체계적 접근을 선호
- 감정보다는 사실과 데이터에 기반한 해결책 요구

ENFP (활동가) 유형 특성:
- 열정적이고 창의적인 성격의 소유자
- 사람들과의 관계를 중요시하며 공감 능력이 뛰어남
- 새로운 가능성과 아이디어에 흥미를 보임
- 불만 상황에서도 긍정적 해결책을 모색
- 개인적 가치관이 존중받기를 원함

ISTJ (논리주의자) 유형 특성:  
- 체계적이고 신뢰할 수 있는 성격
- 기존 규칙과 절차를 중요시함
- 안정성과 예측 가능성을 선호
- 명확하고 구체적인 해결 방법을 원함
- 단계별로 차근차근 진행하는 것을 좋아함

ESFJ (집정관) 유형 특성:
- 타인을 돕고 조화를 이루려는 성향이 강함
- 감정적 지지와 격려를 중요시함  
- 구체적이고 실용적인 도움을 선호
- 개인적 관심과 배려를 통한 해결 방식 선호
- 공동체와 관계 중심의 접근을 원함

ENTP (변론가) 유형 특성:
- 혁신적이고 유연한 사고를 가진 아이디어뱅크
- 새로운 도전과 변화를 즐김
- 다양한 관점에서 문제를 바라봄
- 창의적이고 독창적인 해결책을 선호
- 토론과 브레인스토밍을 통한 문제 해결

ISFP (모험가) 유형 특성:
- 개인의 가치와 신념을 중요시하는 예술가 기질
- 진정성과 authenticity를 추구
- 개인적이고 맞춤형 접근을 선호
- 감정적 공감과 이해를 통한 소통
- 자신만의 속도로 문제 해결 진행

ESTJ (경영자) 유형 특성:
- 목표 지향적이고 실용적인 리더십 스타일
- 효율성과 생산성을 중요시함
- 명확한 계획과 실행을 선호
- 즉각적인 행동과 결과를 중시
- 체계적이고 조직적인 접근 방식

INFJ (옹호자) 유형 특성:
- 깊이 있는 통찰력과 직관적 이해력
- 장기적 비전과 의미 있는 목적 추구
- 근본적 원인 파악과 본질적 해결책 선호
- 신중하고 사려깊은 접근 방식
- 개인의 성장과 발달에 관심

심리 상담 기법:
1. 적극적 경청: 고객의 말을 끝까지 들어주기
2. 공감적 반응: 고객의 감정을 이해하고 공감 표현
3. 명료화: 고객의 문제를 명확히 정리해주기
4. 재구성: 문제를 다른 관점에서 바라보도록 도움
5. 해결책 탐색: 함께 실현 가능한 해결방안 모색

스트레스 대처 방법:
- INTJ: 혼자만의 시간, 체계적 계획 수립
- ENFP: 사회적 지지, 창의적 활동
- ISTJ: 일상 루틴 유지, 단계적 접근
- ESFJ: 타인과의 대화, 실질적 도움
- ENTP: 새로운 도전, 다양한 시각 탐색
- ISFP: 개인적 성찰, 예술적 표현
- ESTJ: 목표 설정, 즉각적 행동
- INFJ: 깊이 있는 사고, 의미 탐구

불만 처리 심리학:
- 첫인상의 중요성: 처음 3분이 전체 상담 분위기 결정
- 감정 완화 기법: 고객의 부정적 감정을 먼저 수용하고 안정화
- 해결 지향적 접근: 문제 중심이 아닌 해결책 중심으로 대화 진행
- 맞춤형 커뮤니케이션: 개인의 성격에 따른 차별화된 접근
- 신뢰 관계 구축: 지속적인 관심과 일관된 태도 유지
"""

# 벡터 스토어 초기화 함수
def initialize_vector_store():
    """심리학 전문 지식을 벡터 데이터베이스로 구축하거나 불러오기"""
    try:
        # 벡터 스토어 디렉토리가 있는지 확인
        if os.path.exists(FAISS_INDEX_DIR):
            st.info("✅ 기존 벡터 스토어를 불러오는 중...")
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            st.success("🎉 벡터 스토어가 성공적으로 로드되었습니다!")
            return vector_store
        else:
            st.info("🔄 새 벡터 스토어를 생성 중입니다...")
            # 텍스트 분할기 설정
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ":", "-", " "]
            )
            
            # 텍스트를 청크로 분할
            chunks = text_splitter.split_text(PSYCHOLOGY_KNOWLEDGE)
            
            # OpenAI 임베딩 모델 초기화
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # FAISS 벡터 스토어 생성
            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings,
                metadatas=[{"source": f"psychology_kb_chunk_{i}"} for i in range(len(chunks))]
            )
            
            # 벡터 스토어 저장
            vector_store.save_local(FAISS_INDEX_DIR)
            st.success("🎉 새 벡터 스토어가 생성 및 저장되었습니다!")
            return vector_store
            
    except Exception as e:
        st.error(f"벡터 스토어 초기화/로딩 실패: {str(e)}")
        return None

# 성격 분석 함수 (RAG 기반으로 강화)
def analyze_personality_with_rag(user_input, vector_store):
    """RAG를 활용한 고도화된 성격 분석"""
    
    # 기본 키워드 매칭
    for mbti_type, info in MBTI_STYLES.items():
        for keyword in info["keywords"]:
            if keyword in user_input:
                # 벡터 스토어에서 해당 MBTI 관련 정보 검색
                if vector_store:
                    docs = vector_store.similarity_search(f"{mbti_type} 특성", k=2)
                    context = "\n".join([doc.page_content for doc in docs])
                    return mbti_type, context
                return mbti_type, ""
    
    # 고급 패턴 분석
    patterns = {
        "INTJ": ["왜", "어떻게", "방법", "해결", "분석", "시스템"],
        "ENFP": ["느낌", "생각", "아이디어", "가능성", "흥미", "재미"],
        "ISTJ": ["절차", "방법", "단계", "정확", "확실", "안전"],
        "ESFJ": ["기분", "마음", "속상", "도와", "관계", "사람들"],
        "ENTP": ["새로운", "다른", "변화", "실험", "도전", "혁신"],
        "ISFP": ["개인", "나만", "특별", "의미", "가치", "중요"],
        "ESTJ": ["빨리", "바로", "즉시", "효과", "결과", "성과"],
        "INFJ": ["이해", "깊이", "본질", "미래", "의미", "통찰"]
    }
    
    for mbti_type, keywords in patterns.items():
        if any(word in user_input for word in keywords):
            if vector_store:
                docs = vector_store.similarity_search(f"{mbti_type} 특성", k=2)
                context = "\n".join([doc.page_content for doc in docs])
                return mbti_type, context
            return mbti_type, ""
    
    return "ENFP", ""  # 기본값

# RAG 기반 응답 생성 함수
def generate_rag_response(user_input, personality_type, vector_store, chat_history=None):
    """RAG를 활용한 전문적인 심리 상담 응답 생성"""
    
    style_info = MBTI_STYLES.get(personality_type, MBTI_STYLES["ENFP"])
    
    # 벡터 스토어에서 관련 문서 검색
    relevant_docs = ""
    if vector_store:
        # 사용자 질문과 성격 유형에 기반한 검색
        search_query = f"{user_input} {personality_type} 상담"
        docs = vector_store.similarity_search(search_query, k=3)
        relevant_docs = "\n".join([doc.page_content for doc in docs])
    
    # RAG 기반 프롬프트 템플릿
    system_prompt = f"""당신은 MBTI 심리학 전문가이자 경험이 풍부한 심리 상담사입니다.

[고객 성격 유형 분석]
- 성격 유형: {personality_type}
- 커뮤니케이션 스타일: {style_info['style']}
- 선호하는 톤: {style_info['tone']}
- 접근 방식: {style_info['approach']}

[전문 심리학 지식베이스]
{relevant_docs}

[상담 지침]
1. 위 전문 지식을 바탕으로 정확하고 전문적인 조언 제공
2. 고객의 성격 유형에 최적화된 커뮤니케이션 스타일 사용
3. 구체적이고 실현 가능한 해결책 제시
4. 공감과 이해를 바탕으로 한 따뜻한 상담
5. 필요시 전문적인 심리학 개념과 기법 활용

한국어로 자연스럽고 전문적으로 답변해주세요."""

    if chat_history:
        messages = [("system", system_prompt)]
        for msg in chat_history[-6:]:  # 최근 6개 메시지만 포함
            messages.append((msg["role"], msg["content"]))
        messages.append(("human", "{input}"))
    else:
        messages = [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template

# 파일 업로드 및 벡터 스토어 업데이트 함수
def update_vector_store_with_file(uploaded_file, vector_store):
    """업로드된 파일로 벡터 스토어 업데이트"""
    try:
        if uploaded_file.type == "application/pdf":
            # PDF 처리
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # 파일 정리
            os.unlink(tmp_file_path)
            
        else:
            # 텍스트 파일 처리
            content = uploaded_file.read().decode("utf-8")
            documents = [{"page_content": content, "metadata": {"source": uploaded_file.name}}]
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        texts = []
        metadatas = []
        
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content if hasattr(doc, 'page_content') else doc["page_content"])
            texts.extend(chunks)
            metadatas.extend([{"source": uploaded_file.name + f"_chunk_{i}"} for i in range(len(chunks))])
        
        # 기존 벡터 스토어에 추가
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        new_vectors = FAISS.from_texts(texts, embeddings, metadatas)
        
        # 벡터 스토어 병합
        vector_store.merge_from(new_vectors)
        
        # 변경 사항을 파일에 저장
        vector_store.save_local(FAISS_INDEX_DIR)
        
        return True, f"{uploaded_file.name} 파일이 성공적으로 추가되었습니다."
        
    except Exception as e:
        return False, f"파일 처리 중 오류 발생: {str(e)}"

# 메인 UI 함수
def main():
    st.title(" 심리와 MBTI 전문 AI 상담사 (RAG 기반)")
    st.markdown("### 전문 심리학 지식을 바탕으로 한 맞춤형 상담 서비스")
    st.markdown("---")
    
    # 벡터 스토어 초기화 (하드에서 불러오거나 새로 생성)
    if "vector_store" not in st.session_state:
        with st.spinner("전문 심리학 지식베이스를 로딩중입니다..."):
            st.session_state.vector_store = initialize_vector_store()
    
    # 사이드바 설정
    with st.sidebar:
        st.header(" 상담 스타일 설정")
        
        # 파일 업로드 기능
        st.subheader("📄 자료 추가")
        uploaded_file = st.file_uploader(
            "심리학 또는 상담스타일 관련 문서를 업로드하세요 (PDF, TXT)",
            type=['pdf', 'txt'],
            help="업로드한 자료는 상담에 활용됩니다"
        )
        
        if uploaded_file and st.button("📤 문서 추가"):
            success, message = update_vector_store_with_file(uploaded_file, st.session_state.vector_store)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.divider()
        
        # MBTI 수동 선택
        mbti_type = st.selectbox(
            "MBTI 유형 선택:",
            ["자동 분석"] + list(MBTI_STYLES.keys()),
            help="자동 분석을 선택하면 대화 내용으로 분석합니다"
        )
        
        # 상담 모드 선택
        counseling_mode = st.radio(
            "상담 모드:",
            ["일반 상담", "불만 처리", "스트레스 관리", "관계 상담"],
            help="상담 주제에 따라 접근 방식이 달라집니다"
        )
        
        # 현재 감지된 성격 유형 표시
        if "current_mbti" in st.session_state and "mbti_context" in st.session_state:
            st.success(f"🎯 분석된 유형: **{st.session_state.current_mbti}**")
            
            with st.expander("📋 성격 분석 정보"):
                style = MBTI_STYLES[st.session_state.current_mbti]
                st.write(f"**스타일:** {style['style']}")
                st.write(f"**톤:** {style['tone']}")
                st.write(f"**접근법:** {style['approach']}")
                
                if st.session_state.mbti_context:
                    st.write("**전문 지식 기반 분석:**")
                    st.write(st.session_state.mbti_context[:200] + "...")
        
        # 대화 초기화
        if st.button("🔄 상담 초기화"):
            st.session_state.messages = []
            if "current_mbti" in st.session_state:
                del st.session_state.current_mbti
            if "mbti_context" in st.session_state:
                del st.session_state.mbti_context
            st.rerun()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 이전 대화 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if user_input := st.chat_input("마음 편히 말씀해주세요..."):
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 성격 유형 분석
        if mbti_type == "자동 분석":
            detected_type, context = analyze_personality_with_rag(
                user_input, 
                st.session_state.vector_store
            )
            st.session_state.current_mbti = detected_type
            st.session_state.mbti_context = context
        else:
            detected_type = mbti_type
            st.session_state.current_mbti = detected_type
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            try:
                # OpenAI 모델 초기화
                llm = ChatOpenAI(
                    temperature=0.7,
                    model="gpt-3.5-turbo",
                    max_tokens=1500
                )
                
                # RAG 기반 프롬프트 생성
                prompt_template = generate_rag_response(
                    user_input, 
                    detected_type, 
                    st.session_state.vector_store,
                    st.session_state.messages
                )
                
                # 응답 생성
                with st.spinner("전문적인 상담을 준비하고 있습니다..."):
                    chain = prompt_template | llm | StrOutputParser()
                    response = chain.invoke({"input": user_input})
                
                # 전문성 표시와 함께 응답
                full_response = f"{response}\n\n---\n* **{detected_type} 유형 맞춤 상담** | 📚 전문 심리학 지식 기반 | 🎯 {counseling_mode} 모드*"
                
                st.markdown(full_response)
                
                # 응답 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })
                
            except Exception as e:
                st.error(f"❌ 상담 중 오류가 발생했습니다: {str(e)}")
                st.info("💡 API 키 설정이나 네트워크 연결을 확인해주세요.")

    # 하단 정보 패널
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔍 RAG 기반 전문 상담의 특징"):
            st.markdown("""
            **🎯 전문성 강화 요소:**
            - 심리학 전문 지식베이스 활용
            - MBTI 유형별 맞춤 상담 기법
            - 실시간 관련 문서 검색
            - 상황별 최적화된 조언 제공
            
            **📚 지식 데이터베이스:**
            - MBTI 8가지 유형 특성 분석
            - 심리 상담 전문 기법  
            - 스트레스 대처 방법론
            - 불만 처리 심리학 이론
            """)
    
    with col2:
        with st.expander("📖 MBTI 유형별 상담 특징"):
            st.markdown("""
            **🧠 분석가형 (NT)**
            - INTJ: 논리적 체계적 접근
            - ENTP: 혁신적 유연한 사고
            
            **🎭 외교관형 (NF)**  
            - ENFP: 창의적 열정적 지지
            - INFJ: 깊이있는 통찰적 조언
            
            **⚖️ 관리자형 (SJ)**
            - ISTJ: 명확한 단계적 가이드
            - ESFJ: 따뜻한 공감적 케어
            
            **🎨 탐험가형 (SP)**
            - ESTJ: 효율적 실용적 해결
            - ISFP: 개인적 진정성있는 지지
            """)

if __name__ == "__main__":
    main()