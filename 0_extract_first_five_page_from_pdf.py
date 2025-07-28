import argparse
import logging
import os
import re
from typing import Set

import pandas as pd
from dotenv import load_dotenv
from openai import Client, OpenAI
from openai.types import FileObject
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)
from pypdf import PdfReader, PdfWriter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()

dirPath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="Parse PDF files and return needed part for CSRone."
)
parser.add_argument(
    "--pdfs",
    type=str,
    default=os.path.join(dirPath, "../docs"),
    help="Directory path to original PDF files.",
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(dirPath, "../outputs"),
    help="Directory path to output PDF files.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="""
    <identity>
        You are an expert AI assistant specialized in analyzing PDF document structures, specifically focusing on extracting information from Tables of Contents (TOC) and identifying standard report sections. You excel at identifying text patterns (in Traditional Chinese), understanding complex page numbering schemes (including covers, TOC pages, Roman/Arabic numerals, and offsets between listed and actual PDF page numbers), and precisely calculating page ranges based on document structure for removal. You pay meticulous attention to identifying the full extent of appendix sections based *only* on the provided information.
    </identity>
    <purpose>
        To meticulously identify specific sections (Cover, Table of Contents, introductory messages like "經營者的話", report information like "關於本報告書", contact details, and the *entire contiguous block* of Appendices) from the first 5 pages of a provided PDF document, calculate their corresponding actual page ranges within the PDF, and output these ranges consolidated and formatted for removal. The goal is to identify *all* pages belonging to these sections, especially ensuring the *complete* appendix block is captured if its boundaries are discernible within the input.
    </purpose>
    <context>
        - The input consists of the first 5 pages of a PDF document.
        - These initial pages typically contain the Cover Page (封面), potentially blank pages, and the Table of Contents (目錄).
        - The TOC lists sections and their corresponding starting page numbers. Page numbering in the TOC might use Roman numerals (i, ii, iii) for introductory sections and Arabic numerals (1, 2, 3) for the main content, or just Arabic numerals throughout.
        - A critical task is to determine the offset between page numbers *listed* in the TOC and the *actual physical page numbers* within the PDF file (where the cover is always actual page 1).
        - Appendices ("附錄") often appear towards the end of the document and may consist of multiple subsections (e.g., "附錄 I", "附錄 II"). The objective is to identify the *entire block* from the start of the first appendix item to the end of the last appendix item listed before a different type of section begins.
        - The target sections/pages to identify for removal are:
            - 封面 (Cover Page - assumed to be actual page 1).
            - 目錄 (The Table of Contents pages themselves).
            - Introductory sections like "董事長/總經理的話" (Chairman/President's Message) or "經營者的話" (Management's Message).
            - Report information sections like "關於報告書" (About the Report) or "關於本報告書" (About This Report).
            - "聯絡我們" (Contact Us).
            - **附錄 (Appendices):** The entire contiguous block of pages starting from the *first* appendix entry found in the TOC until the beginning of the *next non-appendix section* listed immediately after the appendix block.
        - The desired output is a consolidated list of the *actual* PDF page ranges for *all* these target sections/pages.
    </context>
    <task>
        You must perform the following steps rigorously, applying Chain-of-Thought reasoning internally:
        1.  **Analyze Input:** Examine the provided 5 PDF pages.
        2.  **Identify Cover:** The Cover Page is actual PDF page 1. Add `1` to the removal list.
        3.  **Identify TOC Location & Determine Offset:**
            *   Locate the actual page number(s) where the Table of Contents (目錄) resides.
            *   Find the *first section listed in the TOC that has an Arabic page number* (e.g., '1'). Note this TOC page number (`first_arabic_toc_page`).
            *   Determine the *actual PDF page number* where the content for `first_arabic_toc_page` begins. This might require careful scanning of the first 5 pages or inferring based on the TOC structure (e.g., if TOC page '1' is listed on actual page 3, assume the content starts on actual page 4 if page 3 only contains TOC entries). Let this be `actual_page_for_toc_page_1`.
            *   Calculate the offset: `offset = actual_page_for_toc_page_1 - 1`. This offset applies to all *Arabic* page numbers listed in the TOC.
            *   Determine the actual page range occupied by the TOC itself. This starts from the first actual page containing "目錄" and ends on the actual page *just before* `actual_page_for_toc_page_1`. Add this range to the removal list.
        4.  **Identify Standard Sections & TOC Ranges:** Scan the TOC (within the first 5 pages) for:
            *   "董事長/總經理的話" or "經營者的話"
            *   "關於報告書" or "關於本報告書"
            *   "聯絡我們"
        5.  **Calculate Actual Ranges for Standard Sections:** For each standard section found:
            *   Note its starting page number listed in the TOC (`toc_start`). Handle Roman numerals appropriately if they precede the main numbering; they are usually part of the introductory/TOC pages already captured. If the section uses an Arabic number:
            *   Find the starting TOC page number of the *immediately following* section listed in the TOC (`toc_next_start`).
            *   If `toc_next_start` is found within the first 5 pages, the TOC range is `toc_start` to `toc_next_start - 1`.
            *   Calculate the actual PDF range: `actual_start = toc_start + offset`, `actual_end = (toc_next_start - 1) + offset`. Add this range to the removal list.
            *   If `toc_next_start` is *not* found within the first 5 pages, we cannot determine the end page. Do not add a range for this section unless its `actual_start` page is already included in the Cover or TOC ranges.
        6.  **Identify Appendix Block & Calculate Actual Range:**
            *   Scan the TOC (within the first 5 pages) to find the starting TOC page number of the *first* section identified as an Appendix (e.g., containing "附錄", "Appendix", "Supplement"). Let this be `appendix_toc_start`.
            *   There must have an appendix found in every PDF, please make sure you have got the page range from the TOC before output the page range, if you did not find one, please try to scan the end of the TOC again, there's where the appendix should usually be placed.
            *   The page number may not be placed directly beside the title "附錄", so you may want to scan the sections below the appendix block to find the page range of the appendix.
            *   If an appendix start (`appendix_toc_start`) is found, diligently scan the *subsequent* entries in the TOC *within the first 5 pages* to find the starting TOC page number of the *very first section listed AFTER the LAST appendix item that is clearly NOT part of the appendix block*. This signifies the end of the appendices. Let this be `next_non_appendix_start`. Look for entries that lack terms like "附錄" or common appendix indicators and represent distinct content (e.g., "Index", "Company Information", "Assurance Statement", "GRI Content Index" if not explicitly labeled as an appendix itself).
        7.  **Consolidate Ranges:** Collect all identified actual page numbers and ranges (Cover, TOC, standard sections, the complete appendix block if its full range was determined). Merge any ranges that overlap or are directly adjacent (e.g., `1`, `2-3`, `4-6` becomes `1-6`). Sort the final ranges.
        8.  **Format Output:** Output the final, consolidated list of actual PDF page ranges using the exact format `1-2, 5, 6-8`.
    </task>
    <constraints>
        - Process *only* the information contained within the provided first 5 pages of the PDF.
        - Base all calculations strictly on the content and structure observed within these 5 pages.
        - The Cover is always actual page 1.
        - Offset calculation must be based on the first *Arabic* numbered page listed in the TOC and its corresponding actual page.
        - Only include ranges for the specified target sections.
        - The final output *must* be only the comma-separated list of pages/ranges (e.g., `1-4, 8, 50-64`).
        - ***Do not include any introductory text, explanations, comments, apologies, or any text other than the formatted page ranges.***
    </constraints>
    <examples>
        <example>
            <input_description>Assume the first 5 pages of a PDF are provided.
            - Page 1: Cover page text.
            - Page 2: "目錄" title. Lists: "經營者的話 ...... i".
            - Page 3: Continues "目錄". Lists: "關於本報告書 .... 1", "永續亮點 ....... 5". Shows page number 'i' at the bottom.
            - Page 4: Continues "目錄". Lists: "... 利害關係人溝通 ... 48", "附錄 I: GRI .... 50". Shows page number 'ii' at the bottom.
            - Page 5: Continues "目錄". Lists: "附錄 II: SASB ... 55", "附錄 III: TCFD .. 60", "獨立確信報告書 ... 65". Shows page number 'iii' at the bottom.
            </input_description>
            <reasoning>
            1. Analyze: 5 pages.
            2. Cover: Add `1`. RL: [1].
            3. TOC & Offset: TOC starts actual page 2. First Arabic TOC page is '1' ("關於本報告書"). Assume content starts actual page 4. Offset = 4 - 1 = +3. TOC actual range = 2-3. Add `2-3`. RL: [1, 2-3].
            4. Standard Sections: "經營者的話" (TOC 'i', covered by 2-3). "關於本報告書" (TOC 1). "聯絡我們" (not listed).
            5. Actual Ranges (Standard):
               - "關於本報告書": TOC 1. Next is "永續亮點" (TOC 5). Range 1-4. Actual = (1+3) to (4+3) = 4-7. Add `4-7`. RL: [1, 2-3, 4-7].
            6. Appendix Block: First appendix is "附錄 I: GRI", starts TOC page 50. The last listed appendix is "附錄 III: TCFD" (TOC 60). The *first non-appendix section listed after this* is "獨立確信報告書", starting TOC page 65. *Both* start of first appendix (50) AND start of next non-appendix (65) are visible. Appendix block TOC range = 50 to (65-1) = 50-64. Actual range = (50+3) to (64+3) = 53-67. Add `53-67`. RL: [1, 2-3, 4-7, 53-67].
            7. Consolidate: [1, 2-3, 4-7, 53-67]. Merge 1, 2-3, 4-7 -> `1-7`. Final list: [1-7, 53-67].
            8. Format Output.
            </reasoning>
            <output>1-7, 53-67</output>
        </example>
        <example>
            <input_description>Assume the first 5 pages of a PDF are provided.
            - Page 1: Cover.
            - Page 2: TOC Start. Lists: "關於報告書 .... 1".
            - Page 3: TOC Cont. Lists: "總經理的話 .... 5".
            - Page 4: TOC Cont. Lists: "... Chapter X ... 45", "附錄 A: Data ... 50".
            - Page 5: TOC Cont. Lists: "附錄 B: Sources . 55". (End of provided pages, no further sections listed).
            </input_description>
            <reasoning>
            1. Analyze: 5 pages.
            2. Cover: Add `1`. RL: [1].
            3. TOC & Offset: TOC starts p2. First Arabic TOC is '1'. Assume content starts actual p3. Offset = 3 - 1 = +2. TOC actual range is page 2. Add `2`. RL: [1, 2].
            4. Standard Sections: "關於報告書" (TOC 1), "總經理的話" (TOC 5).
            5. Actual Ranges (Standard):
               - "關於報告書": TOC 1. Next is "總經理的話" (TOC 5). Range 1-4. Actual = (1+2) to (4+2) = 3-6. Add `3-6`. RL: [1, 2, 3-6].
               - "總經理的話": TOC 5. Next is "Chapter X" (TOC 45). Range 5-44. Actual = (5+2) to (44+2) = 7-46. Add `7-46`. RL: [1, 2, 3-6, 7-46].
            6. Appendix Block: First appendix "附錄 A" starts TOC 50. The last listed item on page 5 is "附錄 B". *No section listed after this appendix item is visible* within the first 5 pages. Therefore, `next_non_appendix_start` cannot be determined. Per the constraint, the full appendix block's range cannot be confirmed from the input. *Do not add an appendix range*.
            7. Consolidate: [1, 2, 3-6, 7-46]. Merge 1, 2 -> `1-2`. Merge 3-6, 7-46 -> `3-46`. Final list: [1-2, 3-46].
            8. Format Output.
            </reasoning>
            <output>1-2, 3-46</output>
        </example>
    </examples>
    """,
    help="Prompt for OpenAI.",
)
parser.add_argument(
    "--assistant-description",
    type=str,
    default="""
    You are an expert AI assistant specialized in analyzing PDF document structures, specifically
    focusing on extracting information from Tables of Contents (TOC) and identifying standard report
    sections.""",
    help="Assistant model for OpenAI.",
)
args = parser.parse_args()


def load_pdf_files(path: str) -> list[str]:
    if not os.path.isdir(args.output):
        logger.error(f"Directory not found: {path}")
        return []

    return [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if filename.lower().endswith(".pdf")
    ]


def parse_pdf_file(file_path: str) -> str | None:
    temp_pdf = os.path.join(dirPath, "../temp/processing.pdf")
    extract_first_five_pages(temp_pdf, PdfReader(file_path), PdfWriter())

    client, pdf_assistant, thread, file = setup_openai(temp_pdf)
    create_message(client, thread.id, file.id)
    run = send_message(client, thread.id, pdf_assistant.id)

    if run.status != "completed":
        logger.error("Failed to get response from OpenAI.")
        return None

    return get_result(client, thread.id)


def extract_all_number_formats(text):
    pattern = r"\b(\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)*)\b"
    matches = re.findall(pattern, text)

    return matches[0]


def parse_page_ranges(page_range_str: str) -> Set[int]:
    """
    Parse a string of page ranges like "1-2, 5, 6-8" into a set of page numbers.

    Args:
        page_range_str: String containing page ranges (e.g., "1-2, 5, 6-8")

    Returns:
        Set of page numbers to remove (0-indexed)
    """
    pages_to_remove = set()
    parts = [p.strip() for p in page_range_str.split(",")]

    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            pages_to_remove.update(range(int(start) - 1, int(end)))
        else:
            pages_to_remove.add(int(part) - 1)

    return pages_to_remove


def remove_pages_from_pdf(
    pages_to_remove: Set[int], input_path: str, output_path: str
) -> bool:
    """
    Remove specified pages from a PDF and save the result to a new file.

    Args:
        pages_to_remove: Set of page numbers to remove (0-indexed)
        input_path: Path to the input PDF file
        output_path: Path where the output PDF will be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()

        for i, page in enumerate(reader.pages):
            if i not in pages_to_remove:
                writer.add_page(page)

        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        logger.info(f"Successfully created PDF with {len(writer.pages)} pages")
        logger.info(f"Removed pages: {sorted([p + 1 for p in pages_to_remove])}")
        return True

    except Exception as e:
        logger.error(f"Error removing pages from PDF: {str(e)}")
        return False


def add_result_to_df(
    df: pd.DataFrame, pdf: str, pages_to_remove: Set[int]
) -> pd.DataFrame:
    """
    Add the result of the page removal to the DataFrame.

    Args:
        df: DataFrame to update
        pdf: Path to the original PDF file
        pages_to_remove: Set of page numbers removed (0-indexed)

    Returns:
        Updated DataFrame
    """
    pdf_name = os.path.basename(pdf)
    match = re.search(r"\(中\)([\u4e00-\u9fff-a-zA-Z]+)(\d+)", pdf_name)
    if match:
        name = match.group(1)
        stock = match.group(2)
    else:
        # Fallback: try to extract name and stock with a looser pattern
        alt_match = re.search(r"([\u4e00-\u9fff-a-zA-Z]+)(\d+)", pdf_name)
        if alt_match:
            name = alt_match.group(1)
            stock = alt_match.group(2)
        else:
            name = pdf_name
            stock = ""
    removed_pages = sorted([p + 1 for p in pages_to_remove])

    temp_df = pd.DataFrame(
        {
            "name": [name],
            "stock": [stock],
            "removed_pages": [removed_pages],
            "removed_count": [len(removed_pages)],
        }
    )
    return pd.concat([df, temp_df], ignore_index=True)


def extract_first_five_pages(
    pdf: str, pdf_reader: PdfReader, pdf_writer: PdfWriter
) -> None:
    for i in range(5):
        pdf_writer.add_page(pdf_reader.pages[i])
        with open(pdf, "wb"):
            pdf_writer.write(pdf)


def setup_openai(pdf: str) -> tuple[Client, Assistant, Thread, FileObject]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    pdf_assistant = get_pdf_assistant(client)

    thread = client.beta.threads.create()
    file = client.files.create(file=open(pdf, "rb"), purpose="assistants")

    return client, pdf_assistant, thread, file


def get_pdf_assistant(client: OpenAI):
    return client.beta.assistants.create(
        model="gpt-4o",
        description=args.assistant_description,
        tools=[{"type": "file_search"}],
        name="pdf-assistant",
    )


def create_message(client: OpenAI, thread_id: str, file_id: str):
    attachment = Attachment(
        file_id=file_id, tools=[AttachmentToolFileSearch(type="file_search")]
    )

    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=args.prompt, attachments=[attachment]
    )


def send_message(
    client: OpenAI, thread_id: str, assistant_id: str, timeout: int = 1000
) -> Run:
    return client.beta.threads.runs.create_and_poll(
        thread_id=thread_id, assistant_id=assistant_id, timeout=timeout
    )


def get_result(client: OpenAI, thread_id: str) -> str:
    message_cursor = client.beta.threads.messages.list(thread_id=thread_id)
    messages = [message for message in message_cursor]
    return messages[0].content[0].text.value


if __name__ == "__main__":
    pdfs_source = os.path.abspath(args.pdfs)
    output_dir = os.path.abspath(args.output)

    os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame(
        columns=["name", "stock", "removed_pages", "removed_count"]
    )
    pdfs = load_pdf_files(pdfs_source)
    for pdf in pdfs:
        result = parse_pdf_file(pdf)
        result = extract_all_number_formats(result)
        pages_to_remove = parse_page_ranges(result)
        result_df = add_result_to_df(result_df, pdf, pages_to_remove)
        remove_pages_from_pdf(
            pages_to_remove, pdf, os.path.join(str(output_dir), str(os.path.basename(pdf)))
        )
    output_file_dir = os.path.join(str(output_dir), "result.csv")
    result_df.to_csv(output_file_dir, index=False)
