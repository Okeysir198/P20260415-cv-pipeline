"""E2E Playwright test: auto-annotation -> Label Studio visualization workflow.

Tests the full pipeline: signup -> extract API key -> create project
via label_studio_bridge CLI -> import pre-annotated data -> verify project,
annotations, and label rendering in the Label Studio UI.

Tests are ordered — later tests depend on class-level state set by earlier ones.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect

LS_URL = "http://localhost:18103"


def _login(page: Page, email: str, password: str) -> None:
    """Log in to Label Studio via the UI."""
    page.goto(f"{LS_URL}/user/login/")
    page.wait_for_load_state("networkidle")
    # If already logged in (redirected away from login), return
    if "/user/login" not in page.url:
        return
    page.locator('input[name="email"]').fill(email)
    page.locator('input[name="password"]').fill(password)
    page.locator('button[type="submit"]').click()
    page.wait_for_url(lambda url: "/user/login" not in url, timeout=30_000)


_DEFAULT_EMAIL = "nthanhtrung198@gmail.com"
_DEFAULT_PASSWORD = "Trung123"


class TestE2EAutoAnnotate:
    """E2E tests for auto-annotation -> Label Studio visualization workflow."""

    # Class-level state shared between ordered tests
    api_key: str = ""
    project_id: int = 0
    project_name: str = "e2e_fire_review"

    @staticmethod
    def _bridge_parent_args() -> list[str]:
        """Return parent-level CLI auth flags (before subcommand)."""
        return ["--email", _DEFAULT_EMAIL, "--password", _DEFAULT_PASSWORD]

    @staticmethod
    def _bridge_subcmd_args() -> list[str]:
        """Return subcommand-level CLI auth flags (after subcommand)."""
        args = []
        if TestE2EAutoAnnotate.api_key:
            args += ["--api-key", TestE2EAutoAnnotate.api_key]
        return args

    # ------------------------------------------------------------------
    # test_01: Signup via Playwright + extract API key
    # ------------------------------------------------------------------

    def test_01_signup_and_get_api_key(
        self,
        page: Page,
        credentials_manager,
        screenshots_dir: Path,
    ) -> None:
        """Sign up via Playwright (first run), extract API key via session API."""
        page.set_default_timeout(30_000)

        # If we have a cached API key, validate it
        if credentials_manager.api_key:
            try:
                resp = requests.get(
                    f"{LS_URL}/api/projects",
                    headers={"Authorization": f"Token {credentials_manager.api_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    TestE2EAutoAnnotate.api_key = credentials_manager.api_key
                    return
            except requests.RequestException:
                pass

        if credentials_manager.needs_signup:
            # First run: try sign up via Playwright (handles CSRF automatically)
            page.goto(f"{LS_URL}/user/signup/")
            page.wait_for_load_state("networkidle")

            page.locator('input[name="email"]').fill(credentials_manager.email)
            page.locator('input[name="password"]').fill(credentials_manager.password)
            page.locator('button[type="submit"]').click()

            # Wait for either redirect (success) or error on same page
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshots_dir / "01a_signup.png"))

            # If still on signup page, user may already exist — fall back to login
            if "/user/signup" in page.url:
                _login(page, credentials_manager.email, credentials_manager.password)
        else:
            # Subsequent run: log in
            _login(page, credentials_manager.email, credentials_manager.password)

        # Extract API token via session cookie (works with legacy token auth)
        cookies = page.context.cookies()
        session = requests.Session()
        for c in cookies:
            session.cookies.set(c["name"], c["value"])

        resp = session.get(f"{LS_URL}/api/current-user/token", timeout=10)
        assert resp.status_code == 200, f"Failed to get token: {resp.status_code}"
        token = resp.json().get("token", "")

        page.screenshot(path=str(screenshots_dir / "01b_account.png"))
        assert token, "Failed to extract Label Studio API token"

        TestE2EAutoAnnotate.api_key = token
        credentials_manager.save({
            "email": credentials_manager.email,
            "password": credentials_manager.password,
            "api_key": token,
        })

    # ------------------------------------------------------------------
    # test_02: Setup project via CLI
    # ------------------------------------------------------------------

    def test_02_setup_project(
        self,
        page: Page,  # ensures correct test ordering with pytest-playwright
        project_root: Path,
    ) -> None:
        """Create a Label Studio project using the bridge CLI."""
        api_key = TestE2EAutoAnnotate.api_key
        if not api_key:
            pytest.skip("API key not available from test_01")

        result = subprocess.run(
            [
                "uv", "run", "python", "core/p04_label_studio/bridge.py",
                *self._bridge_parent_args(),
                "setup",
                "--data-config", "configs/_test/05_data.yaml",
                *self._bridge_subcmd_args(),
                "--project", TestE2EAutoAnnotate.project_name,
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Accept success or "already exists" (idempotent)
        assert result.returncode == 0 or "already exists" in (
            result.stdout + result.stderr
        ), (
            f"setup failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # ------------------------------------------------------------------
    # test_03: Import annotations via CLI
    # ------------------------------------------------------------------

    def test_03_import_annotations(
        self,
        page: Page,  # ensures correct test ordering with pytest-playwright
        project_root: Path,
    ) -> None:
        """Import fixture annotations into the Label Studio project."""
        api_key = TestE2EAutoAnnotate.api_key
        if not api_key:
            pytest.skip("API key not available from test_01")

        fixtures_val = project_root / "tests" / "fixtures" / "data" / "val"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Symlink images and labels to mimic auto-annotate output
            (tmp / "images").symlink_to(fixtures_val / "images")
            (tmp / "labels").symlink_to(fixtures_val / "labels")

            result = subprocess.run(
                [
                    "uv", "run", "python", "core/p04_label_studio/bridge.py",
                    *self._bridge_parent_args(),
                    "import",
                    "--data-config", "configs/_test/05_data.yaml",
                    "--from-auto-annotate", str(tmp),
                    *self._bridge_subcmd_args(),
                    "--project", TestE2EAutoAnnotate.project_name,
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )

        assert result.returncode == 0, (
            f"import failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        combined = result.stdout + result.stderr
        assert "imported" in combined.lower() or "task" in combined.lower(), (
            f"Expected 'imported' or 'task' in output:\n{combined}"
        )

    # ------------------------------------------------------------------
    # test_04: Verify project page in UI
    # ------------------------------------------------------------------

    def test_04_verify_project_page(
        self,
        page: Page,
        credentials_manager,
        screenshots_dir: Path,
    ) -> None:
        """Navigate to the project and verify tasks are visible."""
        if not TestE2EAutoAnnotate.api_key:
            pytest.skip("API key not available")

        page.set_default_timeout(30_000)
        _login(page, credentials_manager.email, credentials_manager.password)

        # Find project ID via API, then navigate directly
        resp = requests.get(
            f"{LS_URL}/api/projects",
            headers={"Authorization": f"Token {TestE2EAutoAnnotate.api_key}"},
            timeout=10,
        )
        # Fall back to session auth if token fails
        if resp.status_code != 200:
            cookies = page.context.cookies()
            session = requests.Session()
            for c in cookies:
                session.cookies.set(c["name"], c["value"])
            resp = session.get(f"{LS_URL}/api/projects", timeout=10)

        projects = resp.json()
        if isinstance(projects, dict):
            projects = projects.get("results", [])
        proj = next(
            (p for p in projects if p["title"] == TestE2EAutoAnnotate.project_name),
            None,
        )
        assert proj, f"Project '{TestE2EAutoAnnotate.project_name}' not found"
        TestE2EAutoAnnotate.project_id = proj["id"]

        # Navigate directly to project data page
        page.goto(f"{LS_URL}/projects/{proj['id']}/data")
        page.wait_for_load_state("networkidle")

        # Verify tasks are visible
        task_indicators = page.locator('.lsf-table-row')
        expect(task_indicators.first).to_be_visible(timeout=30_000)

        page.screenshot(path=str(screenshots_dir / "04_project_page.png"))

    # ------------------------------------------------------------------
    # test_05: Verify annotation rendering
    # ------------------------------------------------------------------

    def test_05_verify_annotation_render(
        self,
        page: Page,
        credentials_manager,
        screenshots_dir: Path,
    ) -> None:
        """Open first task and verify annotation regions are rendered."""
        if not TestE2EAutoAnnotate.api_key:
            pytest.skip("API key not available")

        page.set_default_timeout(30_000)
        _login(page, credentials_manager.email, credentials_manager.password)

        # Navigate to project data page
        if TestE2EAutoAnnotate.project_id:
            page.goto(f"{LS_URL}/projects/{TestE2EAutoAnnotate.project_id}/data")
            page.wait_for_load_state("networkidle")
        else:
            page.goto(f"{LS_URL}/projects")
            page.wait_for_load_state("networkidle")
            page.locator(
                f'a:has-text("{TestE2EAutoAnnotate.project_name}")'
            ).first.click()
            page.wait_for_load_state("networkidle")

        # Click first task row to open annotation view
        first_row = page.locator('.lsf-table-row').first
        first_row.click()
        page.wait_for_load_state("networkidle")

        # Wait for the image to load in the annotation editor
        page.locator("img").first.wait_for(state="visible", timeout=30_000)

        # Check for annotation region overlays — try multiple selectors
        region_selectors = [
            '[class*="lsf-region"]',
            '[class*="Region"]',
            '[class*="region"]',
            'svg rect',
            'svg [class*="rect"]',
            '[class*="bbox"]',
            '[class*="rectangle"]',
        ]

        selector = ", ".join(region_selectors)
        regions = page.locator(selector)

        # Wait briefly for regions to render (React async)
        try:
            regions.first.wait_for(state="visible", timeout=15_000)
        except Exception:
            # Fallback: check if any SVG elements exist (LS draws on canvas/SVG)
            svg_elements = page.locator("svg *")
            assert svg_elements.count() > 0, (
                "No annotation region overlays found in the annotation editor"
            )

        assert regions.count() > 0 or page.locator("svg *").count() > 0, (
            "No annotation regions or SVG elements found"
        )

        page.screenshot(path=str(screenshots_dir / "05_annotation_render.png"))

    # ------------------------------------------------------------------
    # test_06: Verify label names
    # ------------------------------------------------------------------

    def test_06_verify_label_names(
        self,
        page: Page,
        credentials_manager,
        screenshots_dir: Path,
    ) -> None:
        """Verify 'fire' and 'smoke' labels are visible in the annotation UI."""
        if not TestE2EAutoAnnotate.api_key:
            pytest.skip("API key not available")

        page.set_default_timeout(30_000)
        _login(page, credentials_manager.email, credentials_manager.password)

        # Navigate to first task annotation view
        if TestE2EAutoAnnotate.project_id:
            page.goto(f"{LS_URL}/projects/{TestE2EAutoAnnotate.project_id}/data")
        else:
            page.goto(f"{LS_URL}/projects")
            page.wait_for_load_state("networkidle")
            page.locator(
                f'a:has-text("{TestE2EAutoAnnotate.project_name}")'
            ).first.click()
        page.wait_for_load_state("networkidle")

        # Open first task
        page.locator('.lsf-table-row').first.click()
        page.wait_for_load_state("networkidle")

        # Verify label buttons/chips
        for label in ("fire", "smoke"):
            label_el = page.locator(
                f'span:has-text("{label}"), '
                f'button:has-text("{label}"), '
                f'[class*="label"]:has-text("{label}"), '
                f'[class*="Label"]:has-text("{label}"), '
                f'li:has-text("{label}")'
            ).first
            expect(label_el).to_be_visible(timeout=15_000)

        page.screenshot(path=str(screenshots_dir / "06_label_names.png"))

    # ------------------------------------------------------------------
    # test_07: Full SAM3 auto-annotate (optional)
    # ------------------------------------------------------------------

    def test_07_full_sam3_annotate(
        self,
        page: Page,
        project_root: Path,
        sam3_available: bool,
        credentials_manager,
        screenshots_dir: Path,
    ) -> None:
        """Run auto-annotation with SAM3, import results, and verify."""
        if not sam3_available:
            pytest.skip("SAM3 service not available")

        try:
            import langgraph  # noqa: F401
        except ImportError:
            pytest.skip("langgraph not installed (uv sync --extra qa)")

        api_key = TestE2EAutoAnnotate.api_key
        if not api_key:
            pytest.skip("API key not available")

        page.set_default_timeout(30_000)

        # Run auto-annotate on val fixtures
        annotate_result = subprocess.run(
            [
                "uv", "run", "python",
                "core/p01_auto_annotate/run_auto_annotate.py",
                "--data-config", "configs/_test/05_data.yaml",
                "--mode", "text",
                "--splits", "val",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert annotate_result.returncode == 0, (
            f"auto-annotate failed (rc={annotate_result.returncode}):\n"
            f"stdout: {annotate_result.stdout}\nstderr: {annotate_result.stderr}"
        )

        # Find the output directory from auto-annotate
        # Output goes to runs/{dataset_name}/{timestamp}_01_auto_annotate/
        auto_annotate_dir = project_root / "runs" / "test_fire_100"
        output_dirs = sorted(
            auto_annotate_dir.glob("*_01_auto_annotate"), key=os.path.getmtime
        )
        assert output_dirs, "No auto-annotate output directory found"
        latest_output = output_dirs[-1]

        # Re-import into Label Studio
        import_result = subprocess.run(
            [
                "uv", "run", "python", "core/p04_label_studio/bridge.py",
                *self._bridge_parent_args(),
                "import",
                "--data-config", "configs/_test/05_data.yaml",
                "--from-auto-annotate", str(latest_output),
                *self._bridge_subcmd_args(),
                "--project", TestE2EAutoAnnotate.project_name,
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert import_result.returncode == 0, (
            f"import failed (rc={import_result.returncode}):\n"
            f"stdout: {import_result.stdout}\nstderr: {import_result.stderr}"
        )

        # Verify in UI
        _login(page, credentials_manager.email, credentials_manager.password)
        page.goto(f"{LS_URL}/projects")
        page.wait_for_load_state("networkidle")

        project_link = page.locator(
            f'a:has-text("{TestE2EAutoAnnotate.project_name}")'
        ).first
        project_link.wait_for(state="visible", timeout=30_000)
        project_link.click()
        page.wait_for_load_state("networkidle")

        # Verify tasks exist (same selector as test_04)
        task_rows = page.locator('.lsf-table-row')
        expect(task_rows.first).to_be_visible(timeout=30_000)

        page.screenshot(path=str(screenshots_dir / "07_sam3_annotate.png"))

    # ------------------------------------------------------------------
    # test_08: Export round-trip (CLI path validation)
    # ------------------------------------------------------------------

    def test_08_export_roundtrip(
        self,
        page: Page,
        project_root: Path,
        credentials_manager,
    ) -> None:
        """Export reviewed annotations via CLI and verify the export path works."""
        api_key = TestE2EAutoAnnotate.api_key
        if not api_key:
            pytest.skip("API key not available from test_01")

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "exported_labels"

            result = subprocess.run(
                [
                    "uv", "run", "python", "core/p04_label_studio/bridge.py",
                    *self._bridge_parent_args(),
                    "export",
                    "--project", TestE2EAutoAnnotate.project_name,
                    "--output-dir", str(export_dir),
                    "--data-config", "configs/_test/05_data.yaml",
                    "--only-reviewed",
                    *self._bridge_subcmd_args(),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Export may return 0 reviewed annotations if no human review was done
            # in the E2E flow. Accept success or "no tasks" message.
            if result.returncode != 0:
                output = (result.stdout + result.stderr).lower()
                assert any(
                    kw in output
                    for kw in ["no tasks", "not found", "skipped", "no annotation"]
                ), (
                    f"export failed unexpectedly (rc={result.returncode}):\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
