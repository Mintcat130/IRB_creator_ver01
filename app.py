import streamlit as st
import anthropic
import PyPDF2
import io
import requests
from scholarly import scholarly
from Bio import Entrez
import json
import re
import uuid

#연구계획서 ID 생성
def generate_research_id():
    return str(uuid.uuid4())

#세션 상태 초기화
def reset_session_state():
    keys_to_keep = ['api_key', 'anthropic_client']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.current_research_id = generate_research_id()
    st.session_state.section_contents = {}

# 섹션 내용 저장
def save_section_content(section, content):
    if 'research_data' not in st.session_state:
        st.session_state.research_data = {}
    if st.session_state.current_research_id not in st.session_state.research_data:
        st.session_state.research_data[st.session_state.current_research_id] = {}
    st.session_state.research_data[st.session_state.current_research_id][section] = content

# 섹션 내용 로드
def load_section_content(section):
    if 'research_data' in st.session_state and st.session_state.current_research_id in st.session_state.research_data:
        return st.session_state.research_data[st.session_state.current_research_id].get(section, "")
    return ""


# 페이지 설정을 코드 최상단에 추가
st.set_page_config(page_title="📖연구계획서 작성 도우미", page_icon="📖")

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 병리학 분야의 연구 전문가로서 행동하는 AI 조수입니다. 당신의 역할은 사용자가 연구계획서를 작성하는 데 도움을 주는 것입니다. 사용자는 연구계획서의 특정 항목에 대한 정보를 제공할 것이며, 당신은 이를 바탕으로 해당 항목을 작성해야 합니다.

사용자가 제공한 정보를 주의 깊게 분석하고, 당신의 병리학 연구에 대한 전문 지식을 활용하여 요청된 연구계획서 섹션을 작성하세요. 다음 지침을 따르세요:

1. 사용자가 제공한 정보를 최대한 활용하세요.
2. 필요한 경우, 병리학 연구에 대한 당신의 지식을 사용하여 정보를 보완하세요.
3. 연구계획서 섹션의 구조와 형식을 적절히 유지하세요.
4. 명확하고 전문적인 언어를 사용하세요.
5. 필요한 경우 적절한 참고문헌이나 인용을 포함하세요.

한국어로 작성하되 의학 용어나 통계용어는 괄호 안에 영어 원문을 포함시키세요. 한국어로 번역이 불가능한 고유명사는 영어 그대로 적으세요. 예를 들어, "엽상종양(Phyllodes tumor)", "Student T-검정(Student T-test)"과 같은 형식으로 작성하세요.
"""

# PREDEFINED_PROMPTS 수정
PREDEFINED_PROMPTS = {
    "2. 연구 목적": """
    사용자가 제공한 연구 주제와 키워드를 바탕으로, 연구 목적과 가설을 1000자 이내의 줄글로 작성하세요. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
    다음 사항을 포함해야 합니다:
    1. 연구의 주요 목적
    2. 연구로 인해 의도하는 가설
    3. 가설을 입증하기 위한 구체적인 설명
    4. 연구의 중요성과 예상되는 결과

    사용자 입력:
    {user_input}

    위의 내용을 바탕으로 연구 목적과 가설을 구체화하여 작성해주세요.
    """,
    "3. 연구 배경": """
    제공된 정보를 바탕으로 연구의 배경을 1500자 이내로 설명해주세요. 어미는 반드시 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
    다음 사항을 포함하세요:
    1. 이론적 배경 및 근거
    2. 선행 연구 및 결과
    3. 연구 배경과 연구의 정당성에 대한 설명
    4. 국내외 연구 현황

    사용자 입력:
    {user_input}

    연구 목적:
    {research_purpose}

    검색된 논문:
    {papers}

    PDF 내용:
    {pdf_content}

    위의 내용을 바탕으로 연구 배경을 구체화하여 작성해주세요. 특히 모든 PDF 파일의 내용을 적극적으로 활용하여 연구 배경 작성에 참고해주세요. 참고문헌을 인용할 때는 [저자, 연도] 형식으로 표기해주세요.
    """,

    "4. 선정기준, 제외기준": """
    2, 3번 섹션의 결과물과 참고한 논문들을 토대로, 이 연구에 적당한 대상자 그룹(선정기준)과 연구에서 제외해야 할 그룹(제외기준)을 추천해주세요. 다음 지침을 따라주세요:
    1. 구체적인 년도나 시기는 적지 않습니다. (잘못된 예시: 2009년 국가 건강검진을 받은 4,234,415명)
    2. 선정기준 예시: 40세에서 60세 사이에 해당하며, 이전 치매에 진단받은 과거력이 없는 수검자
    3. 제외기준 예시: 40세 이하 혹은 60세 이상, 검진 당시 치매 진단 과거력 있는 수검자, 누락된 변수 정보가 있는 수검자
    4. 이외 다른 말은 하지 말것.

    연구 목적:
    {research_purpose}

    연구 배경:
    {research_background}

    위의 내용을 바탕으로 적절한 선정기준, 제외기준을 작성해주세요.
    """,
    "5. 대상자 수 및 산출근거": """
이전 섹션의 내용과 업로드된 논문들을 참고하여 다음 형식에 맟춰 대상자 수 및 산출근거를 작성해주세요, 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다):

1) 대상자 수: [숫자]명

2) 산출 근거: 
[여기에 산출 근거를 자세히 설명해주세요. 다음 사항을 포함하세요:]
- 선행연구와 통계학적 평가방법에 근거한 설명
- 가능한 경우, 구체적인 통계적 방법(예: 검정력 분석)을 언급하고 사용된 가정들을 설명
- 대상자 수가 연구 목적을 달성하기에 충분한 이유를 설명

연구 목적:
{research_purpose}

연구 배경:
{research_background}

선정기준, 제외기준:
{selection_criteria}

위의 내용을 바탕으로 적절한 대상자 수와 그 산출근거를 작성해주세요.
""",
        "6. 자료분석과 통계적 방법": """
    이전 섹션의 내용을 바탕으로 자료분석과 통계적 방법을 1500자 이내로 작성해주세요. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다) 다음 사항을 포함해야 합니다:

    1. 수집해야 하는 수치나 값, 변수들 제시
    2. 변수의 이름은 영어로 작성하고, 긴 변수명의 경우 연구에 사용할 수 있는 약자도 함께 제시
    3. 연구에 사용할 군(group) 제시
    4. 연구를 통해 수집된 자료 또는 정보를 이용하는 방법(통계적 방법 포함) 기술
    5. 통계분석(계획) 제시:
       - 통계분석 방법
       - 분석대상군
       - 결측치의 처리 방법
       - 혼란변수 통제방법
       - 유의수준
       - 결과제시와 결과 도출 방안

    연구 목적:
    {research_purpose}

    연구 배경:
    {research_background}

    선정기준, 제외기준:
    {selection_criteria}

    대상자 수 및 산출근거:
    {sample_size}

    위의 내용을 바탕으로 자료분석과 통계적 방법을 구체적으로 작성해주세요. 각 항목을 명확히 구분하여 작성하되, 전체적으로 일관성 있는 내용이 되도록 해주세요.
    """,
        "7. 연구방법": """
    2번부터 6번까지의 섹션 내용을 바탕으로 전체 연구방법을 1000자 이내로 요약해주세요. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다) 다음 사항을 포함해야 합니다:

    1. 연구 목적의 핵심
    2. 연구 대상자 선정 및 제외 기준의 요점
    3. 대상자 수와 그 근거의 간략한 설명
    4. 주요 자료수집 방법
    5. 핵심적인 통계분석 방법

    이 연구가 어떤 방법으로 진행되는지 간단명료하게 설명해주세요. 전문적이면서도 이해하기 쉽게 작성해주세요.

    연구 목적:
    {research_purpose}

    연구 배경:
    {research_background}

    선정기준, 제외기준:
    {selection_criteria}

    대상자 수 및 산출근거:
    {sample_size}

    자료분석과 통계적 방법:
    {data_analysis}

    위의 내용을 바탕으로 전체 연구방법을 요약해주세요.
    """
}


# 연구 섹션 순서 정의
RESEARCH_SECTIONS = [
    "2. 연구 목적",
    "3. 연구 배경",
    "4. 선정기준, 제외기준",
    "5. 대상자 수 및 산출근거",
    "6. 자료분석과 통계적 방법",
    "7. 연구방법",
    # 다른 섹션들은 나중에 추가할 예정입니다.
]

# Anthropic API 클라이언트 초기화 함수
def initialize_anthropic_client(api_key):
    try:
        client = anthropic.Client(api_key=api_key)
        # 간단한 API 호출로 키 유효성 검사
        client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            messages=[{"role": "user", "content": "Hello"}]
        )
        return client
    except Exception as e:
        st.error(f"API 키 초기화 중 오류 발생: {str(e)}")
        return None

#세션 초기화 함수
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.clear()

                
#AI 응답 생성 함수
def generate_ai_response(prompt):
    if 'anthropic_client' in st.session_state and st.session_state.anthropic_client:
        try:
            system_prompt = f"{SYSTEM_PROMPT}\n\n추가 지시사항: 답변을 작성할 때 번호나 불렛 포인트를 사용하지 말고, 서술형으로 작성해주세요. 문단을 나누어 가독성 있게 작성하되, 전체적으로 하나의 연결된 글이 되도록 해주세요."
            
            response = st.session_state.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except anthropic.APIError as e:
            st.error(f"Anthropic API 오류: {str(e)}")
            return f"AI 응답 생성 중 API 오류가 발생했습니다: {str(e)}"
        except Exception as e:
            st.error(f"예상치 못한 오류 발생: {str(e)}")
            return f"AI 응답을 생성하는 중 예상치 못한 오류가 발생했습니다: {str(e)}"
    else:
        return "API 클라이언트가 초기화되지 않았습니다. API 키를 다시 확인해주세요."


# PDF 파일 업로드 함수
def upload_pdf():
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        return extract_text_from_pdf(uploaded_file)
    return None

# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# PubMed 검색 함수 (수정)
def search_pubmed(query, max_results=10):
    try:
        # 이메일 주소 설정을 제거하거나 더미 값을 사용
        # Entrez.email = "example@example.com"  # 이 줄을 제거하거나 주석 처리

        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        ids = record["IdList"]
        results = []
        for id in ids:
            handle = Entrez.efetch(db="pubmed", id=id, rettype="medline", retmode="text")
            record = Entrez.read(Entrez.parse(handle))
            if record:
                article = record[0]
                title = article.get("TI", "No title")
                year = article.get("DP", "")[:4]
                authors = ", ".join(article.get("AU", []))[:50] + "..." if len(article.get("AU", [])) > 2 else ", ".join(article.get("AU", []))
                link = f"https://pubmed.ncbi.nlm.nih.gov/{id}/"
                results.append({"title": title, "year": year, "authors": authors, "link": link})
            handle.close()
        return results
    except Exception as e:
        st.error(f"PubMed 검색 중 오류 발생: {str(e)}")
        return []

# Google Scholar 검색 함수 수정
def search_google_scholar(query, max_results=10):
    search_query = scholarly.search_pubs(query)
    results = []
    for i, result in enumerate(search_query):
        if i >= max_results:
            break
        try:
            title = result['bib'].get('title', 'No title')
            year = result['bib'].get('pub_year', 'No year')
            authors = result['bib'].get('author', 'No author')
            if isinstance(authors, list):
                authors = ", ".join(authors[:2]) + "..." if len(authors) > 2 else ", ".join(authors)
            link = result.get('pub_url', '#')
            results.append({"title": title, "year": year, "authors": authors, "link": link})
        except AttributeError:
            continue  # 결과를 건너뛰고 다음 결과로 진행
    return results

def write_research_purpose():
    st.markdown("## 2. 연구 목적")
    
    # 히스토리 초기화
    if "2. 연구 목적_history" not in st.session_state:
        st.session_state["2. 연구 목적_history"] = []

    st.markdown("어떤 연구를 계획중인지, 연구에 대한 내용이나 키워드를 형식에 상관없이 자유롭게 입력해주세요. 입력 후 버튼을 누르면 AI 모델이 연구목적에 대한 줄글을 작성 해 줍니다.")
    
    user_input = st.text_area("연구 주제 및 키워드:", height=150)
    
    if st.button("연구 목적 생성"):
        if user_input:
            prompt = PREDEFINED_PROMPTS["2. 연구 목적"].format(user_input=user_input)
            ai_response = generate_ai_response(prompt)
            
            # 현재 내용을 히스토리에 추가
            current_content = load_section_content("2. 연구 목적")
            if current_content:
                st.session_state["2. 연구 목적_history"].append(current_content)
            
            save_section_content("2. 연구 목적", ai_response)
            st.session_state.show_modification_request = False
            st.rerun()
        else:
            st.warning("연구 주제나 키워드를 입력해주세요.")

    # AI 응답 표시
    content = load_section_content("2. 연구 목적")
    if content:
        st.markdown("### AI가 생성한 연구 목적:")
        st.markdown(content)
        
        char_count = len(content)
        st.info(f"생성된 내용의 글자 수: {char_count}/1000")
        
        if char_count > 1000:
            st.warning("생성된 내용이 1000자를 초과했습니다. 수정이 필요할 수 있습니다.")

        # 수정 요청 기능
        if st.button("수정 요청하기"):
            st.session_state.show_modification_request = True
            st.rerun()

        if st.session_state.get('show_modification_request', False):
            modification_request = st.text_area(
                "수정을 원하는 부분과 수정 방향을 설명해주세요:",
                height=150,
                key="modification_request_2"
            )
            if st.button("수정 요청 제출", key="submit_modification_2"):
                if modification_request:
                    current_content = load_section_content("2. 연구 목적")
                    # 현재 내용을 히스토리에 추가
                    st.session_state["2. 연구 목적_history"].append(current_content)
                    
                    prompt = f"""
                    현재 연구 목적:
                    {current_content}

                    사용자의 수정 요청:
                    {modification_request}

                    위의 수정 요청을 반영하여 연구 목적을 수정해주세요. 다음 지침을 따라주세요:
                    1. 전체 내용을 유지하면서, 수정 요청된 부분만 변경하세요.
                    2. 수정 요청에 명시적으로 언급되지 않은 부분은 그대로 유지하세요.
                    3. 수정된 내용은 자연스럽게 기존 내용과 연결되어야 합니다.
                    4. 전체 내용은 1000자를 넘지 않아야 합니다.
                    5. 수정된 부분은 기존 내용의 맥락과 일관성을 유지해야 합니다.
                    7. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
                    8. 내용 이외 다른말은 하지 말것.
                    
                    수정된 전체 연구 목적을 작성해주세요.
                    """
                    modified_response = generate_ai_response(prompt)
                    
                    save_section_content("2. 연구 목적", modified_response)
                    st.session_state.show_modification_request = False
                    st.rerun()
                else:
                    st.warning("수정 요청 내용을 입력해주세요.")

    # 편집 기능
    edited_content = st.text_area(
        "생성된 내용을 편집하세요:",
        content,
        height=300,
        key="edit_content_2"
    )
    st.warning("다음 섹션으로 넘어가기 전에 편집내용 저장 버튼을 누르세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("편집 내용 저장", key="save_edit_2"):
            # 현재 내용을 히스토리에 추가
            st.session_state["2. 연구 목적_history"].append(content)
            save_section_content("2. 연구 목적", edited_content)
            st.success("편집된 내용이 저장되었습니다.")
            st.rerun()
    with col2:
        if st.button("실행 취소", key="undo_edit_2"):
            if st.session_state["2. 연구 목적_history"]:
                # 히스토리에서 마지막 항목을 가져와 현재 내용으로 설정
                previous_content = st.session_state["2. 연구 목적_history"].pop()
                save_section_content("2. 연구 목적", previous_content)
                st.success("이전 버전으로 되돌렸습니다.")
                st.rerun()
            else:
                st.warning("더 이상 되돌릴 수 있는 버전이 없습니다.")


# 연구 배경 작성 함수 (수정)
def write_research_background():
    st.markdown("## 3. 연구 배경")

    # 히스토리 초기화
    if "3. 연구 배경_history" not in st.session_state:
        st.session_state["3. 연구 배경_history"] = []
    
    # 키워드 입력
    keywords = st.text_input("연구 배경 작성을 위한 참조논문 검색에 사용할 키워드를 입력하세요 (최대 10개, 쉼표로 구분):")
    keywords_list = [k.strip() for k in keywords.split(',') if k.strip()][:10]
    
    if keywords_list:
        st.write("입력된 키워드:", ", ".join(keywords_list))
        
    if st.button("논문 검색"):
        if keywords_list:
            search_query = " ".join(keywords_list)
            
            with st.spinner("논문을 검색 중입니다..."):
                pubmed_results = search_pubmed(search_query)
                scholar_results = search_google_scholar(search_query)
            
            st.session_state.pubmed_results = pubmed_results
            st.session_state.scholar_results = scholar_results
            st.success("검색이 완료되었습니다.")
            st.rerun()
            
    # 검색 결과 표시
    if 'pubmed_results' in st.session_state:
        st.subheader("PubMed 검색 결과")
        for i, result in enumerate(st.session_state.pubmed_results):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"[{result['title']} ({result['year']})]({result['link']})")
                st.caption(f"저자: {result['authors']}")
            with col2:
                if st.button("삭제", key=f"del_pubmed_{i}"):
                    del st.session_state.pubmed_results[i]
                    st.rerun()
    
    if 'scholar_results' in st.session_state:
        st.subheader("Google Scholar 검색 결과")
        for i, result in enumerate(st.session_state.scholar_results):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"[{result['title']} ({result['year']})]({result['link']})")
                st.caption(f"저자: {result['authors']}")
            with col2:
                if st.button("삭제", key=f"del_scholar_{i}"):
                    del st.session_state.scholar_results[i]
                    st.rerun()
            
    # PDF 파일 업로드
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요 (여러 개 선택 가능)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.pdf_texts = []
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.session_state.pdf_texts.append(pdf_text)
        st.success(f"{len(uploaded_files)}개의 PDF 파일이 성공적으로 업로드되었습니다.")

    # 연구 배경 생성 버튼
    if st.button("해당 내용으로 연구배경 작성하기"):
        if 'pubmed_results' in st.session_state or 'scholar_results' in st.session_state or 'pdf_texts' in st.session_state:
            research_purpose = load_section_content("2. 연구 목적")
            
            papers = []
            if 'pubmed_results' in st.session_state:
                papers.extend([f"{r['title']} ({r['year']})" for r in st.session_state.pubmed_results])
            if 'scholar_results' in st.session_state:
                papers.extend([f"{r['title']} ({r['year']})" for r in st.session_state.scholar_results])
            papers_text = "\n".join(papers)
            
            pdf_content = "\n".join(st.session_state.get('pdf_texts', []))
            
            prompt = PREDEFINED_PROMPTS["3. 연구 배경"].format(
                user_input=keywords,
                research_purpose=research_purpose,
                papers=papers_text,
                pdf_content=pdf_content
            )
            
            ai_response = generate_ai_response(prompt)
            
            # 현재 내용을 히스토리에 추가
            current_content = load_section_content("3. 연구 배경")
            if current_content:
                st.session_state["3. 연구 배경_history"].append(current_content)
            
            save_section_content("3. 연구 배경", ai_response)
            st.session_state.show_modification_request_3 = False
            st.rerun()
        else:
            st.warning("논문을 검색하거나 PDF를 업로드한 후 다시 시도해주세요.")

    # AI 응답 표시
    content = load_section_content("3. 연구 배경")
    if content:
        st.markdown("### AI가 생성한 연구 배경:")
        st.markdown(content)
        
        char_count = len(content)
        st.info(f"생성된 내용의 글자 수: {char_count}/1500")
        
        if char_count > 1500:
            st.warning("생성된 내용이 1500자를 초과했습니다. 수정이 필요할 수 있습니다.")

        # 수정 요청 기능
        if st.button("수정 요청하기", key="request_modification_3"):
            st.session_state.show_modification_request_3 = True
            st.rerun()

        if st.session_state.get('show_modification_request_3', False):
            modification_request = st.text_area(
                "수정을 원하는 부분과 수정 방향을 설명해주세요:",
                height=150,
                key="modification_request_3"
            )
            if st.button("수정 요청 제출", key="submit_modification_3"):
                if modification_request:
                    current_content = load_section_content("3. 연구 배경")
                    # 현재 내용을 히스토리에 추가
                    st.session_state["3. 연구 배경_history"].append(current_content)
                    
                    prompt = f"""
                    현재 연구 배경:
                    {current_content}

                    사용자의 수정 요청:
                    {modification_request}

                    위의 수정 요청을 반영하여 연구 배경을 수정해주세요. 다음 지침을 따라주세요:
                    1. 전체 내용을 유지하면서, 수정 요청된 부분만 변경하세요.
                    2. 수정 요청에 명시적으로 언급되지 않은 부분은 그대로 유지하세요.
                    3. 수정된 내용은 자연스럽게 기존 내용과 연결되어야 합니다.
                    4. 전체 내용은 1500자를 넘지 않아야 합니다.
                    5. 수정된 부분은 기존 내용의 맥락과 일관성을 유지해야 합니다.
                    6. 연구 배경의 논리적 흐름을 유지하세요.
                    7. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
                    8. 내용 이외 다른말은 하지 말것.
                    
                    수정된 전체 연구 배경을 작성해주세요.  
                    """
                    modified_response = generate_ai_response(prompt)
                    
                    save_section_content("3. 연구 배경", modified_response)
                    st.session_state.show_modification_request_3 = False
                    st.rerun()
                else:
                    st.warning("수정 요청 내용을 입력해주세요.")

    # 편집 기능
    edited_content = st.text_area(
        "생성된 내용을 편집하세요:",
        content,
        height=300,
        key="edit_content_3"
    )
    st.warning("다음 섹션으로 넘어가기 전에 편집내용 저장 버튼을 누르세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("편집 내용 저장", key="save_edit_3"):
            # 현재 내용을 히스토리에 추가
            st.session_state["3. 연구 배경_history"].append(content)
            save_section_content("3. 연구 배경", edited_content)
            st.success("편집된 내용이 저장되었습니다.")
            st.rerun()
    with col2:
        if st.button("실행 취소", key="undo_edit_3"):
            if st.session_state["3. 연구 배경_history"]:
                # 히스토리에서 마지막 항목을 가져와 현재 내용으로 설정
                previous_content = st.session_state["3. 연구 배경_history"].pop()
                save_section_content("3. 연구 배경", previous_content)
                st.success("이전 버전으로 되돌렸습니다.")
                st.rerun()
            else:
                st.warning("더 이상 되돌릴 수 있는 버전이 없습니다.")

# 선정기준, 제외기준 작성 함수
def write_selection_criteria():
    st.markdown("## 4. 선정기준, 제외기준")
    
    # 히스토리 초기화
    if "4. 선정기준, 제외기준_history" not in st.session_state:
        st.session_state["4. 선정기준, 제외기준_history"] = []

    if st.button("선정, 제외기준 AI에게 추천받기"):
        research_purpose = load_section_content("2. 연구 목적")
        research_background = load_section_content("3. 연구 배경")
        
        prompt = PREDEFINED_PROMPTS["4. 선정기준, 제외기준"].format(
            research_purpose=research_purpose,
            research_background=research_background
        )
        
        ai_response = generate_ai_response(prompt)
        
        # 현재 내용을 히스토리에 추가
        current_content = load_section_content("4. 선정기준, 제외기준")
        if current_content:
            st.session_state["4. 선정기준, 제외기준_history"].append(current_content)
        
        save_section_content("4. 선정기준, 제외기준", ai_response)
        st.rerun()

    # AI 응답 표시
    content = load_section_content("4. 선정기준, 제외기준")
    if content:
        st.markdown("### AI가 추천한 선정, 제외기준:")
        st.markdown(content)

        # 수정 요청 기능
        if st.button("수정 요청하기", key="request_modification_4"):
            st.session_state.show_modification_request_4 = True
            st.rerun()

        if st.session_state.get('show_modification_request_4', False):
            modification_request = st.text_area(
                "수정을 원하는 부분과 수정 방향을 설명해주세요:",
                height=150,
                key="modification_request_4"
            )
            if st.button("수정 요청 제출", key="submit_modification_4"):
                if modification_request:
                    current_content = load_section_content("4. 선정기준, 제외기준")
                    # 현재 내용을 히스토리에 추가
                    st.session_state["4. 선정기준, 제외기준_history"].append(current_content)
                    
                    prompt = f"""
                    현재 선정기준, 제외기준:
                    {current_content}

                    사용자의 수정 요청:
                    {modification_request}

                    위의 수정 요청을 반영하여 선정기준, 제외기준을 수정해주세요. 어미는 문어제 반말을 사용하세요.(예시: "~했다.", "~있다.", "~이다.") 다음 지침을 따라주세요:
                    1. 전체 내용을 유지하면서, 수정 요청된 부분만 변경하세요.
                    2. 수정 요청에 명시적으로 언급되지 않은 부분은 그대로 유지하세요.
                    3. 수정된 내용은 자연스럽게 기존 내용과 연결되어야 합니다.
                    4. 수정된 부분은 기존 내용의 맥락과 일관성을 유지해야 합니다.
                    5. 내용 이외 다른말은 하지 말것.
                    
                    수정된 전체 선정기준, 제외기준을 작성해주세요.
                    """
                    modified_response = generate_ai_response(prompt)
                    
                    save_section_content("4. 선정기준, 제외기준", modified_response)
                    st.session_state.show_modification_request_4 = False
                    st.rerun()
                else:
                    st.warning("수정 요청 내용을 입력해주세요.")
    
    # 편집 기능
    edited_content = st.text_area(
        "선정기준, 제외기준을 직접 여기에 작성하거나, 위 버튼을 눌러 AI의 추천을 받으세요. 생성된 내용을 편집하세요:",
        content,
        height=300,
        key="edit_content_4"
    )

    st.warning("다음 섹션으로 넘어가기 전에 편집내용 저장 버튼을 누르세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("편집 내용 저장", key="save_edit_4"):
            # 현재 내용을 히스토리에 추가
            st.session_state["4. 선정기준, 제외기준_history"].append(content)
            save_section_content("4. 선정기준, 제외기준", edited_content)
            st.success("편집된 내용이 저장되었습니다.")
            st.rerun()
    with col2:
        if st.button("실행 취소", key="undo_edit_4"):
            if st.session_state["4. 선정기준, 제외기준_history"]:
                # 히스토리에서 마지막 항목을 가져와 현재 내용으로 설정
                previous_content = st.session_state["4. 선정기준, 제외기준_history"].pop()
                save_section_content("4. 선정기준, 제외기준", previous_content)
                st.success("이전 버전으로 되돌렸습니다.")
                st.rerun()
            else:
                st.warning("더 이상 되돌릴 수 있는 버전이 없습니다.")

# 5. 대상자 수 및 산출근거 작성 함수 (수정)
def write_sample_size():
    st.markdown("## 5. 대상자 수 및 산출근거")
    
    # 히스토리 초기화
    if "5. 대상자 수 및 산출근거_history" not in st.session_state:
        st.session_state["5. 대상자 수 및 산출근거_history"] = []

    if st.button("대상자 수 및 산출근거 AI에게 추천받기"):
        research_purpose = load_section_content("2. 연구 목적")
        research_background = load_section_content("3. 연구 배경")
        selection_criteria = load_section_content("4. 선정기준, 제외기준")
        
        prompt = PREDEFINED_PROMPTS["5. 대상자 수 및 산출근거"].format(
            research_purpose=research_purpose,
            research_background=research_background,
            selection_criteria=selection_criteria
        )
        
        ai_response = generate_ai_response(prompt)
        
        # 현재 내용을 히스토리에 추가
        current_content = load_section_content("5. 대상자 수 및 산출근거")
        if current_content:
            st.session_state["5. 대상자 수 및 산출근거_history"].append(current_content)
        
        save_section_content("5. 대상자 수 및 산출근거", ai_response)
        st.rerun()

    # AI 응답 표시
    content = load_section_content("5. 대상자 수 및 산출근거")
    if content:
        st.markdown("### AI가 추천한 대상자 수 및 산출근거:")
        st.markdown(content)

        # 수정 요청 기능
        if st.button("수정 요청하기", key="request_modification_5"):
            st.session_state.show_modification_request_5 = True
            st.rerun()

        if st.session_state.get('show_modification_request_5', False):
            modification_request = st.text_area(
                "수정을 원하는 부분과 수정 방향을 설명해주세요:",
                height=150,
                key="modification_request_5"
            )
            if st.button("수정 요청 제출", key="submit_modification_5"):
                if modification_request:
                    current_content = load_section_content("5. 대상자 수 및 산출근거")
                    # 현재 내용을 히스토리에 추가
                    st.session_state["5. 대상자 수 및 산출근거_history"].append(current_content)
                    
                    prompt = f"""
                    현재 대상자 수 및 산출근거:
                    {current_content}

                    사용자의 수정 요청:
                    {modification_request}

                    위의 수정 요청을 반영하여 대상자 수 및 산출근거를 수정해주세요. 다음 지침을 따라주세요:
                    1. 전체 내용을 유지하면서, 수정 요청된 부분만 변경하세요.
                    2. 수정 요청에 명시적으로 언급되지 않은 부분은 그대로 유지하세요.
                    3. 수정된 내용은 자연스럽게 기존 내용과 연결되어야 합니다.
                    4. 수정된 부분은 기존 내용의 맥락과 일관성을 유지해야 합니다.
                    5. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
                    6. 내용 이외 다른말은 하지 말것.
                    
                    수정된 전체 대상자 수 및 산출근거를 작성해주세요.
                    """
                    modified_response = generate_ai_response(prompt)
                    
                    save_section_content("5. 대상자 수 및 산출근거", modified_response)
                    st.session_state.show_modification_request_5 = False
                    st.rerun()
                else:
                    st.warning("수정 요청 내용을 입력해주세요.")
    
    # 편집 기능
    edited_content = st.text_area(
        "대상자 수 및 산출근거를 직접 여기에 작성하거나, 위 버튼을 눌러 AI의 추천을 받으세요. 생성된 내용을 편집하세요:",
        content,
        height=300,
        key="edit_content_5"
    )

    st.warning("다음 섹션으로 넘어가기 전에 편집내용 저장 버튼을 누르세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("편집 내용 저장", key="save_edit_5"):
            # 현재 내용을 히스토리에 추가
            st.session_state["5. 대상자 수 및 산출근거_history"].append(content)
            save_section_content("5. 대상자 수 및 산출근거", edited_content)
            st.success("편집된 내용이 저장되었습니다.")
            st.rerun()
    with col2:
        if st.button("실행 취소", key="undo_edit_5"):
            if st.session_state["5. 대상자 수 및 산출근거_history"]:
                # 히스토리에서 마지막 항목을 가져와 현재 내용으로 설정
                previous_content = st.session_state["5. 대상자 수 및 산출근거_history"].pop()
                save_section_content("5. 대상자 수 및 산출근거", previous_content)
                st.success("이전 버전으로 되돌렸습니다.")
                st.rerun()
            else:
                st.warning("더 이상 되돌릴 수 있는 버전이 없습니다.")

#6. 자료분석과 통계적 방법 함수
def write_data_analysis():
    st.markdown("## 6. 자료분석과 통계적 방법")
    
    # 히스토리 초기화
    if "6. 자료분석과 통계적 방법_history" not in st.session_state:
        st.session_state["6. 자료분석과 통계적 방법_history"] = []

    if st.button("자료분석 및 통계방법 AI에게 추천받기"):
        research_purpose = load_section_content("2. 연구 목적")
        research_background = load_section_content("3. 연구 배경")
        selection_criteria = load_section_content("4. 선정기준, 제외기준")
        sample_size = load_section_content("5. 대상자 수 및 산출근거")
        
        prompt = PREDEFINED_PROMPTS["6. 자료분석과 통계적 방법"].format(
            research_purpose=research_purpose,
            research_background=research_background,
            selection_criteria=selection_criteria,
            sample_size=sample_size
        )
        
        ai_response = generate_ai_response(prompt)
        
        # 현재 내용을 히스토리에 추가
        current_content = load_section_content("6. 자료분석과 통계적 방법")
        if current_content:
            st.session_state["6. 자료분석과 통계적 방법_history"].append(current_content)
        
        save_section_content("6. 자료분석과 통계적 방법", ai_response)
        st.rerun()

    # AI 응답 표시
    content = load_section_content("6. 자료분석과 통계적 방법")
    if content:
        st.markdown("### AI가 추천한 자료분석과 통계적 방법:")
        st.markdown(content)

        # 수정 요청 기능
        if st.button("수정 요청하기", key="request_modification_6"):
            st.session_state.show_modification_request_6 = True
            st.rerun()

        if st.session_state.get('show_modification_request_6', False):
            modification_request = st.text_area(
                "수정을 원하는 부분과 수정 방향을 설명해주세요:",
                height=150,
                key="modification_request_6"
            )
            if st.button("수정 요청 제출", key="submit_modification_6"):
                if modification_request:
                    current_content = load_section_content("6. 자료분석과 통계적 방법")
                    # 현재 내용을 히스토리에 추가
                    st.session_state["6. 자료분석과 통계적 방법_history"].append(current_content)
                    
                    prompt = f"""
                    현재 자료분석과 통계적 방법:
                    {current_content}

                    사용자의 수정 요청:
                    {modification_request}

                    위의 수정 요청을 반영하여 자료분석과 통계적 방법을 수정해주세요. 다음 지침을 따라주세요:
                    1. 전체 내용을 유지하면서, 수정 요청된 부분만 변경하세요.
                    2. 수정 요청에 명시적으로 언급되지 않은 부분은 그대로 유지하세요.
                    3. 수정된 내용은 자연스럽게 기존 내용과 연결되어야 합니다.
                    4. 전체 내용은 1500자를 넘지 않아야 합니다.
                    5. 수정된 부분은 기존 내용의 맥락과 일관성을 유지해야 합니다.
                    6. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
                    7. 내용 이외 다른말은 하지 말것.
                    
                    수정된 전체 자료분석과 통계적 방법을 작성해주세요.
                    """
                    modified_response = generate_ai_response(prompt)
                    
                    save_section_content("6. 자료분석과 통계적 방법", modified_response)
                    st.session_state.show_modification_request_6 = False
                    st.rerun()
                else:
                    st.warning("수정 요청 내용을 입력해주세요.")
    
    # 편집 기능
    edited_content = st.text_area(
        "자료분석과 통계적 방법을 직접 여기에 작성하거나, 위 버튼을 눌러 AI의 추천을 받으세요. 생성된 내용을 편집하세요:",
        content,
        height=400,
        key="edit_content_6"
    )

    st.warning("다음 섹션으로 넘어가기 전에 편집내용 저장 버튼을 누르세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("편집 내용 저장", key="save_edit_6"):
            # 현재 내용을 히스토리에 추가
            st.session_state["6. 자료분석과 통계적 방법_history"].append(content)
            save_section_content("6. 자료분석과 통계적 방법", edited_content)
            st.success("편집된 내용이 저장되었습니다.")
            st.rerun()
    with col2:
        if st.button("실행 취소", key="undo_edit_6"):
            if st.session_state["6. 자료분석과 통계적 방법_history"]:
                # 히스토리에서 마지막 항목을 가져와 현재 내용으로 설정
                previous_content = st.session_state["6. 자료분석과 통계적 방법_history"].pop()
                save_section_content("6. 자료분석과 통계적 방법", previous_content)
                st.success("이전 버전으로 되돌렸습니다.")
                st.rerun()
            else:
                st.warning("더 이상 되돌릴 수 있는 버전이 없습니다.")

    # 글자 수 표시
    if content:
        char_count = len(content)
        st.info(f"현재 글자 수: {char_count}/1500")
        if char_count > 1500:
            st.warning("글자 수가 1500자를 초과했습니다. 내용을 줄여주세요.")

#7. 연구방법 정리 함수
def write_research_method():
    st.markdown("## 7. 연구방법")
    
    # 히스토리 초기화
    if "7. 연구방법_history" not in st.session_state:
        st.session_state["7. 연구방법_history"] = []

    if st.button("연구방법 정리 요청하기"):
        research_purpose = load_section_content("2. 연구 목적")
        research_background = load_section_content("3. 연구 배경")
        selection_criteria = load_section_content("4. 선정기준, 제외기준")
        sample_size = load_section_content("5. 대상자 수 및 산출근거")
        data_analysis = load_section_content("6. 자료분석과 통계적 방법")
        
        prompt = PREDEFINED_PROMPTS["7. 연구방법"].format(
            research_purpose=research_purpose,
            research_background=research_background,
            selection_criteria=selection_criteria,
            sample_size=sample_size,
            data_analysis=data_analysis
        )
        
        ai_response = generate_ai_response(prompt)
        
        # 현재 내용을 히스토리에 추가
        current_content = load_section_content("7. 연구방법")
        if current_content:
            st.session_state["7. 연구방법_history"].append(current_content)
        
        save_section_content("7. 연구방법", ai_response)
        st.rerun()

    # AI 응답 표시
    content = load_section_content("7. 연구방법")
    if content:
        st.markdown("### AI가 정리한 연구방법:")
        st.markdown(content)

        # 수정 요청 기능
        if st.button("수정 요청하기", key="request_modification_7"):
            st.session_state.show_modification_request_7 = True
            st.rerun()

        if st.session_state.get('show_modification_request_7', False):
            modification_request = st.text_area(
                "수정을 원하는 부분과 수정 방향을 설명해주세요:",
                height=150,
                key="modification_request_7"
            )
            if st.button("수정 요청 제출", key="submit_modification_7"):
                if modification_request:
                    current_content = load_section_content("7. 연구방법")
                    # 현재 내용을 히스토리에 추가
                    st.session_state["7. 연구방법_history"].append(current_content)
                    
                    prompt = f"""
                    현재 연구방법:
                    {current_content}

                    사용자의 수정 요청:
                    {modification_request}

                    위의 수정 요청을 반영하여 연구방법을 수정해주세요. 다음 지침을 따라주세요:
                    1. 전체 내용을 유지하면서, 수정 요청된 부분만 변경하세요.
                    2. 수정 요청에 명시적으로 언급되지 않은 부분은 그대로 유지하세요.
                    3. 수정된 내용은 자연스럽게 기존 내용과 연결되어야 합니다.
                    4. 전체 내용은 1000자를 넘지 않아야 합니다.
                    5. 수정된 부분은 기존 내용의 맥락과 일관성을 유지해야 합니다.
                    6. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
                    7. 내용 이외 다른말은 하지 말것.
                    
                    수정된 전체 연구방법을 작성해주세요.
                    """
                    modified_response = generate_ai_response(prompt)
                    
                    save_section_content("7. 연구방법", modified_response)
                    st.session_state.show_modification_request_7 = False
                    st.rerun()
                else:
                    st.warning("수정 요청 내용을 입력해주세요.")
    
    # 편집 기능
    edited_content = st.text_area(
        "연구방법을 직접 여기에 작성하거나, 위 버튼을 눌러 AI의 정리를 받으세요. 생성된 내용을 편집하세요:",
        content,
        height=400,
        key="edit_content_7"
    )

    st.warning("다음 섹션으로 넘어가기 전에 편집내용 저장 버튼을 누르세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("편집 내용 저장", key="save_edit_7"):
            # 현재 내용을 히스토리에 추가
            st.session_state["7. 연구방법_history"].append(content)
            save_section_content("7. 연구방법", edited_content)
            st.success("편집된 내용이 저장되었습니다.")
            st.rerun()
    with col2:
        if st.button("실행 취소", key="undo_edit_7"):
            if st.session_state["7. 연구방법_history"]:
                # 히스토리에서 마지막 항목을 가져와 현재 내용으로 설정
                previous_content = st.session_state["7. 연구방법_history"].pop()
                save_section_content("7. 연구방법", previous_content)
                st.success("이전 버전으로 되돌렸습니다.")
                st.rerun()
            else:
                st.warning("더 이상 되돌릴 수 있는 버전이 없습니다.")

    # 글자 수 표시
    if content:
        char_count = len(content)
        st.info(f"현재 글자 수: {char_count}/1000")
        if char_count > 1000:
            st.warning("글자 수가 1000자를 초과했습니다. 내용을 줄여주세요.")


def extract_references(text):
    # [저자, 연도] 형식의 참고문헌을 추출
    references = re.findall(r'\[([^\]]+)\]', text)
    return list(set(references))  # 중복 제거

# 여기에 chat_interface 함수가 이어집니다.
def chat_interface():
    st.subheader("IRB 연구계획서 작성 도우미✏️ ver.01 (by HJY)")

    if 'current_research_id' not in st.session_state:
        st.session_state.current_research_id = generate_research_id()

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
        
        if st.button("API 키 확인"):
            client = initialize_anthropic_client(api_key)
            if client:
                st.success("유효한 API 키입니다. 연구계획서 작성하기 버튼을 눌러 시작하세요.")
                st.session_state.temp_api_key = api_key
            else:
                st.error("API 키 설정에 실패했습니다. 키를 다시 확인해 주세요.")
        
        if st.button("연구계획서 작성하기✏️"):
            if 'temp_api_key' in st.session_state:
                st.session_state.api_key = st.session_state.temp_api_key
                st.session_state.anthropic_client = initialize_anthropic_client(st.session_state.api_key)
                del st.session_state.temp_api_key
                st.success("API 키가 설정되었습니다!")
                st.rerun()
            else:
                st.warning("먼저 API 키를 입력하고 확인해주세요.")
    else:
        if 'current_section' not in st.session_state:
            st.session_state.current_section = 'home'

        st.sidebar.text(f"현재 API 키: {st.session_state.api_key[:5]}...")

        if st.sidebar.button("🏠홈으로"):
            st.session_state.current_section = 'home'
            st.rerun()

        if st.sidebar.button("새 연구계획서 시작"):
            reset_session_state()
            st.success("새로운 연구계획서를 시작합니다.")
            st.rerun()

        # 홈 화면 표시
        if st.session_state.current_section == 'home':
            st.markdown("## 연구계획서 작성을 시작합니다")
            st.markdown("아래 버튼을 클릭하여 각 섹션을 작성하세요.")
            
            for section in RESEARCH_SECTIONS:
                if st.button(f"{section} 작성하기"):
                    st.session_state.current_section = section
                    st.rerun()

        else:
            # 현재 섹션에 따른 작성 인터페이스 표시
            if st.session_state.current_section == "2. 연구 목적":
                write_research_purpose()
            elif st.session_state.current_section == "3. 연구 배경":
                write_research_background()
            elif st.session_state.current_section == "4. 선정기준, 제외기준":
                write_selection_criteria()
            elif st.session_state.current_section == "5. 대상자 수 및 산출근거":
                write_sample_size()
            elif st.session_state.current_section == "6. 자료분석과 통계적 방법":
                write_data_analysis()
            elif st.session_state.current_section == "7. 연구방법":
                write_research_method()

            # 이전 섹션과 다음 섹션 버튼
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⬅️이전 섹션"):
                    current_index = RESEARCH_SECTIONS.index(st.session_state.current_section)
                    if current_index > 0:
                        st.session_state.current_section = RESEARCH_SECTIONS[current_index - 1]
                    else:
                        st.session_state.current_section = 'home'
                    st.rerun()

            with col2:
                if st.button("다음 섹션➡️"):
                    current_index = RESEARCH_SECTIONS.index(st.session_state.current_section)
                    if current_index < len(RESEARCH_SECTIONS) - 1:
                        st.session_state.current_section = RESEARCH_SECTIONS[current_index + 1]
                        st.rerun()
                    else:
                        st.success("모든 섹션을 완료했습니다!")

        # 전체 내용 미리보기
        if st.sidebar.button("전체 내용 미리보기"):
            for section in RESEARCH_SECTIONS:
                st.markdown(f"### {section}")
                st.markdown(load_section_content(section))
            
            # 참고문헌 표시
            references = st.session_state.get('references', [])
            if references:
                st.markdown("### 참고문헌")
                for ref in references:
                    st.markdown(f"- {ref}")

    # CSS 스타일
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 60px;
        white-space: normal;
        word-wrap: break-word;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    chat_interface()

