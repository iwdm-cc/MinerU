import os
import json
import shutil
from pathlib import Path
from multiprocessing import Pool, Manager
from tqdm import tqdm

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


PROCESSED_LOG_PATH = "processed_files.json"


def load_processed_files():
    """
    加载已处理PDF的记录文件，返回一个set
    """
    if os.path.exists(PROCESSED_LOG_PATH):
        with open(PROCESSED_LOG_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_files(processed_set):
    """
    保存已处理PDF的文件名到json文件
    """
    with open(PROCESSED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_set)), f, ensure_ascii=False, indent=2)


def process_single_pdf(args):
    pdf_path, output_root = args
    try:
        name_without_extension = pdf_path.stem
        local_image_dir = os.path.join(output_root, name_without_extension, "images")
        local_md_dir = os.path.join(output_root, name_without_extension)
        image_dir = os.path.basename(local_image_dir)

        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)

        markdown_collect_dir = os.path.join(output_root, "markdowns")
        os.makedirs(markdown_collect_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(str(pdf_path))
        ds = PymuDocDataset(pdf_bytes)

        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_extension}_layout.pdf"))
        pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_extension}_spans.pdf"))

        md_file_name = f"{name_without_extension}.md"
        pipe_result.dump_md(md_writer, md_file_name, image_dir)

        # 拷贝到集中目录
        src_md_path = os.path.join(local_md_dir, md_file_name)
        dst_md_path = os.path.join(markdown_collect_dir, md_file_name)
        if os.path.exists(src_md_path):
            shutil.copyfile(src_md_path, dst_md_path)

        pipe_result.dump_content_list(md_writer, f"{name_without_extension}_content_list.json", image_dir)
        pipe_result.dump_middle_json(md_writer, f"{name_without_extension}_middle.json")

        return pdf_path.name  # 成功返回文件名
    except Exception as e:
        print(f"[✗] 处理失败：{pdf_path.name}，错误信息：{e}")
        return None


def batch(pdf_dir, output_dir, max_workers=4):
    """
    多进程批处理，自动记录成功处理的文件
    """
    os.makedirs(output_dir, exist_ok=True)

    all_pdf_paths = [p for p in Path(pdf_dir).glob("*.pdf")]
    processed_files = load_processed_files()

    # 筛选未处理文件
    unprocessed_paths = [p for p in all_pdf_paths if p.name not in processed_files]
    task_args = [(p, output_dir) for p in unprocessed_paths]

    print(f"共找到 PDF 文件 {len(all_pdf_paths)} 个，其中已处理 {len(processed_files)} 个，待处理 {len(task_args)} 个")
    success_files = []

    with Pool(processes=max_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_pdf, task_args), total=len(task_args)):
            if result:
                processed_files.add(result)
                success_files.append(result)
                # 实时写入（可选：降低崩溃风险）
                save_processed_files(processed_files)

    print(f"\n🎉 全部处理完成，新增成功 {len(success_files)} 个")
    save_processed_files(processed_files)


if __name__ == '__main__':
    batch(r"C:\Users\admin\PycharmProjects\MinerU\demo\yuanjian", r"C:\Users\admin\PycharmProjects\MinerU\demo\yuanjian-output",max_workers=2)
