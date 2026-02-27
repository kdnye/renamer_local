import os
import re
import shutil
import asyncio
import sys
import json
import subprocess
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

from ollama import AsyncClient, Client
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InteractiveFileRenamer:
    def __init__(
            self,
            base_dir: Union[str, Path] = None,
            auto_mode: bool = False,
            model_name: str = "qwen2.5-coder:7b",
            context_length: int = 2048,
            filename_prefix: str = "new",
            filename_extension: str = ".txt",
            max_file_name_length: int = 128,
            backup_dir: Union[str, Path] = None,
    ):

        self.base_dir: Path | None = Path(base_dir) if base_dir else None
        self.auto_mode: bool = auto_mode
        self.model_name: str = model_name
        self.context_length: int = context_length
        self.filename_prefix: str = filename_prefix
        self.filename_extension: str = filename_extension
        self.max_file_name_length: int = max_file_name_length
        self.backup_dir: Path = backup_dir
        self.ollama_client: Client = Client()

    def create_backup_directory(self, base_dir: Path) -> Path:
        """
        Create a backup directory within the base directory.
        Ensures unique backup folder names using timestamp or incremental numbering.
        """
        backup_dir = os.path.join(base_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)
        return Path(backup_dir)

    @staticmethod
    def backup_file(
            file_path: Path,
            backup_dir: Path
    ) -> bool:
        """
        Copy a file to the backup directory before renaming.
        Preserves original filename in the backup.
        """
        try:
            shutil.copy2(file_path, backup_dir)
            return True
        except Exception as e:
            logger.error(f"Backup failed for {file_path}: {e}")
            return False

    def _discover_file_patterns(self) -> list[tuple[tuple[str, str], int]]:
        """Discover potential file patterns in the directory."""
        if not self.base_dir or not os.path.isdir(self.base_dir):
            raise ValueError("Invalid directory")
        patterns = Counter()
        for filename in os.listdir(self.base_dir):
            if os.path.isfile(os.path.join(self.base_dir, filename)):
                prefix_match = re.match(r"^([^.\w]{1,5}|\w{1,5})", filename)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    ext = os.path.splitext(filename)[1].lower()
                    patterns[(prefix, ext)] += 1
        return patterns.most_common(5)

    def _interactive_directory_selection(self) -> Path:
        """Interactively select the directory for file renaming."""
        while True:
            dir_path_str = input("Enter the directory path (press Enter for current directory): ").strip()
            dir_path = Path(dir_path_str) if dir_path_str else Path.cwd()
            if dir_path.is_dir():
                return dir_path
            else:
                print(f"Error: '{dir_path}' is not a valid directory. Please try again.")

    def _interactive_file_pattern_selection(
            self,
            patterns: list[tuple[tuple[Any, str], int]]
    ) -> tuple[str, str]:
        """Interactively select file patterns for renaming."""
        print("\nChoose pattern of names for renaming:")
        for i, ((prefix, ext), count) in enumerate(patterns, 1):
            print(f"{i}. `{prefix}` - {count} files, {ext[1:]}")
        print(f"{len(patterns) + 1}. Write your pattern")
        while True:
            try:
                choice = input('Press number of chosen variant: ').strip()
                if not choice:
                    if patterns:
                        return patterns[0][0]
                    else:
                        return "new", ".txt"
                choice = int(choice)
                if 1 <= choice <= len(patterns):
                    return patterns[choice - 1][0]
                elif choice == len(patterns) + 1:
                    prefix = input('Enter file prefix: ').strip()
                    ext = input('Enter file extension (with dot): ').strip().lower()
                    return prefix, ext
                else:
                    print('Invalid selection. Please try again.')
            except ValueError:
                print('Please enter a valid number.')

    def _get_available_ollama_models(self) -> list[str]:
        """Retrieve available Ollama models using native client."""
        try:
            models_data = self.ollama_client.list()
            return [model['name'] for model in models_data.get('models', [])]
        except Exception as e:
            print(f"Error retrieving Ollama models: {e}")
            return []

    def _interactive_model_selection(
            self,
            available_models: list[str]
    ) -> str:
        """Interactively select an Ollama model for renaming."""
        if not available_models:
            print("No Ollama models found. Using default model.")
            return 'qwen2.5-coder:7b'
        print("\nAvailable Ollama models:")
        for i, model in enumerate(available_models, 1):
            logger.info(f"{i}. {model}")
        print(f"{len(available_models) + 1}. Enter custom model name")

        while True:
            try:
                choice = input("Select model number (or press Enter for first model): ").strip()
                if not choice:
                    return available_models[0]
                choice = int(choice)
                if 1 <= choice <= len(available_models):
                    return available_models[choice - 1]
                elif choice == len(available_models) + 1:
                    return input("Enter full model name: ").strip()
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def _get_model_max_context(
            self,
            model_name: str
    ) -> int:
        """Get the maximum context length for a given model."""
        try:
            model_info = self.ollama_client.show(model_name)
            if model_info and model_info.modelinfo:
                context_length = None
                for key, value in model_info.modelinfo.items():
                    if key.endswith(".context_length"):
                        context_length = int(value)
                        break
                if context_length:
                    print(f"Current context length for {model_name} is: {context_length}")
                    return context_length
            print(f"Could not determine context length for {model_name} (key not found)")
            return 2048
        except Exception as e:
            print(f"Could not retrieve model info: {e}")
            return 2048

    def interactive_configuration(self) -> None:
        """Orchestrate interactive configuration."""
        if self.auto_mode:
            return
        self.base_dir = self._interactive_directory_selection()
        patterns = self._discover_file_patterns()
        self.filename_prefix, self.filename_extension = self._interactive_file_pattern_selection(patterns)
        available_models = self._get_available_ollama_models()
        self.model_name = self._interactive_model_selection(available_models)
        max_context = self._get_model_max_context(self.model_name)
        while True:
            context_choice = input(
                f"\nModel max context: {max_context} characters. Enter custom length, `500` will work (or press Enter for max): ").strip()
            if not context_choice:
                self.context_length = max_context
                break
            elif context_choice.isdigit() and int(context_choice) <= max_context:
                self.context_length = int(context_choice)
                break
            else:
                print(f"Invalid context length. Please enter a number less than or equal to {max_context}.")
        print("\n--- Configuration Summary ---")
        print(f"Directory: {self.base_dir}")
        print(f"File Pattern: '{self.filename_prefix}*{self.filename_extension}'")
        print(f"Model: {self.model_name}")
        print(f"Context Length: {self.context_length} characters")

    def _sanitize_filename(
            self,
            suggested_name: str,
            original_ext: str
    ) -> str:
        suggested_name = suggested_name.replace('.', '')
        suggested_name = suggested_name.replace('`', '').replace("'", "")
        forbidden_chars = r'[<>/\\\|\?\*\:\"\+\%\!\@]'
        suggested_name = re.sub(forbidden_chars, "_", suggested_name)
        suggested_name = re.sub(r"\s+", "-", suggested_name)
        suggested_name = re.sub(r"-{2,}", "-", suggested_name)
        suggested_name = re.sub(r"^[^a-zA-Z]+", "", suggested_name)
        suggested_name = re.sub(r"[^a-zA-Z0-9]+$", "", suggested_name)
        if not suggested_name:
            suggested_name = f"untitled-document {uuid.uuid4()}"
        suggested_name = f"{suggested_name}{original_ext}"
        return suggested_name

    async def analyze_and_rename(
            self,
            file_path: Path,
            filename: str
    ) -> tuple[bool, str, str]:
        if self.backup_dir is None:
            self.backup_dir = self.create_backup_directory(self.base_dir)

        backup_success = self.backup_file(file_path, self.backup_dir)
        if not backup_success:
            logger.warning(f"Could not backup {filename} before renaming")
            return False, filename, "Backup failed"
        if not os.path.exists(file_path):
            print(f"File {file_path} no longer exists.")
            return False, filename, "File does not exist"

       try:
            if file_path.suffix.lower() == '.pdf':
                import pypdf
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    content = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
                        if len(content) >= self.context_length:
                            break
                    content = content[:self.context_length]
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(self.context_length)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return False, filename, str(e)

        prompt = f"""
                        The following text is from a file in a software development project:
    
                        ```
                        {content}
                        ```
    
                        Suggest a file name using the following pattern:
    
                        `<Type> - <Action> - <Component> - <Short Description>.<Extension>`
    
                        Where:
                        * **Type:**  task, issue, doc, snippet, review, idea
                        * **Action:** create, fix, update, review, discuss, complete
                        * **Component:** backend, frontend, database, api, auth, testing, or a more specific component name
                        * **Short Description:** A concise summary of the file's content.
                        * **Extension:** Based on the content (e.g., .md, .txt, .py). If unsure, use .md
    
                        Provide only the file name as output, maximum 128 characters, lowercase. 
                        """

        try:
            client = AsyncClient()
            response = await client.generate(
                self.model_name,
                prompt,
                options={
                    "num_predict": self.max_file_name_length,
                }
            )
            suggested_name = response["response"].strip().lower()
        except Exception as e:
            print(f"Error generating new filename for {filename}: {e}")
            return False, filename, str(e)
        original_ext = os.path.splitext(filename)[1]
        suggested_name = re.sub(r"\.\w+$", "", suggested_name)
        new_filename = self._sanitize_filename(suggested_name, original_ext)
        try:
            new_file_path = self.base_dir / new_filename
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
            return True, new_filename, "Success"
        except Exception as e:
            print(f"Error renaming {filename}: {e}")
            return False, filename, str(e)

    def collect_files_to_rename(self) -> list[Path]:
        return [
            file_path
            for file_path in self.base_dir.iterdir()
            if file_path.name.startswith(self.filename_prefix) and file_path.name.endswith(self.filename_extension)
        ]

    async def _sequential_rename(
            self,
            files_to_rename: list[Path]
    ) -> tuple[int, int, list[tuple[str, str]]]:
        renamed_count = 0
        failed_count = 0
        failed_files = []
        for i, file_path in enumerate(files_to_rename, 1):
            filename = os.path.basename(file_path)
            success, result_filename, message = await self.analyze_and_rename(file_path,
                                                                              filename)  # TODO: fix Expected type 'Path', got 'str' instead
            if success:
                renamed_count += 1
            else:
                failed_count += 1
                failed_files.append((filename, message))
            if i % 10 == 0 or i == len(files_to_rename):
                print(f"Progress: {renamed_count} renamed, {failed_count} failed")
        return renamed_count, failed_count, failed_files

    async def _parallel_rename(
            self,
            files_to_rename: list[Path],
            parallel_count: int
    ) -> tuple[int, int, list[tuple[str, str]]]:
        semaphore = asyncio.Semaphore(parallel_count)

        async def rename_with_semaphore(file_path: Path):
            async with semaphore:
                filename = os.path.basename(file_path)
                return await self.analyze_and_rename(file_path, filename)

        results = await asyncio.gather(*[rename_with_semaphore(file_path) for file_path in files_to_rename])
        renamed_count = sum(1 for success, _, _ in results if success)
        failed_count = sum(1 for success, _, _ in results if not success)
        failed_files = [(filename, message) for success, filename, message in zip(
            [r[0] for r in results],
            [os.path.basename(f) for f in files_to_rename],
            [r[2] for r in results]
        ) if not success]
        return renamed_count, failed_count, failed_files

    async def rename_files(
            self,
            parallel_count: int = 0
    ) -> tuple[int, int, list[tuple[str, str]]]:
        files_to_rename = self.collect_files_to_rename()
        if not files_to_rename:
            print("No files found to rename.")
            return 0, 0, []
        print(f"Found {len(files_to_rename)} files to rename")
        if parallel_count > 0:
            return await self._parallel_rename(files_to_rename, parallel_count)
        else:
            return await self._sequential_rename(files_to_rename)


async def main():
    parser = argparse.ArgumentParser(description="Interactive File Renaming Tool")
    parser.add_argument("-a", "--auto", action="store_true",
                        help="Run in auto mode without interactive prompts")
    parser.add_argument("-d", "--directory",
                        help="Specify directory non-interactively")
    parser.add_argument('-m', '--model', help='Specify the Ollama model')
    parser.add_argument('-c', '--context', type=int, help='Specify the context length')
    parser.add_argument('--prefix', help="Specify the file prefix")
    parser.add_argument('--ext', help="Specify the file extension")
    parser.add_argument('-p', '--parallel', type=int, default=0,
                        help='Number of parallel rename operations (default: sequential)')
    args = parser.parse_args()

    try:
        renamer = InteractiveFileRenamer(
            base_dir=args.directory,
            auto_mode=args.auto,
            model_name=args.model,
            context_length=args.context,
            filename_prefix=args.prefix,
            filename_extension=args.ext,
        )
        renamer.interactive_configuration()

        try:
            renamed_count, failed_count, failed_files = await renamer.rename_files(args.parallel)
            print("\n--- Renaming Operation Summary ---")
            print(f"Total files processed: {renamed_count + failed_count}")
            print(f"Successfully renamed: {renamed_count}")
            print(f"Failed to rename: {failed_count}")
            if failed_files:
                print("\nFiles that failed to rename:")
                for file, error in failed_files:
                    print(f"  - {file}: {error}")

        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
        except Exception as e:
            if "ollama" in str(e).lower():
                logger.critical(f"A fatal error occurred during model interaction: {e}. Aborting.")
                sys.exit(1)
            else:
                logger.error(f"An unexpected error occurred: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
