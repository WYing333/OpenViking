#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for TreeBuilder final URI metadata."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestFinalizeFromTemp:
    @staticmethod
    def _make_fs(entries, existing_uris: set[str]):
        fs = MagicMock()

        async def _ls(uri, **kwargs):
            return entries[uri]

        async def _stat(uri, **kwargs):
            if uri in existing_uris:
                return {"name": uri.split("/")[-1], "isDir": True}
            raise FileNotFoundError(f"Not found: {uri}")

        async def _exists(uri, **kwargs):
            return uri in existing_uris

        fs.ls = AsyncMock(side_effect=_ls)
        fs.stat = AsyncMock(side_effect=_stat)
        fs.exists = AsyncMock(side_effect=_exists)
        return fs

    @pytest.mark.asyncio
    async def test_resources_root_to_behaves_like_parent(self):
        from openviking.parse.tree_builder import TreeBuilder
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        entries = {
            "viking://temp/import": [{"name": "tt_b", "isDir": True}],
            "viking://temp/import/tt_b": [
                {"name": "a.md", "isDir": False},
                {"name": "b.md", "isDir": False},
            ],
        }
        fs = self._make_fs(entries, {"viking://resources"})
        builder = TreeBuilder()
        ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

        with patch("openviking.parse.tree_builder.get_viking_fs", return_value=fs):
            tree = await builder.finalize_from_temp(
                temp_dir_path="viking://temp/import",
                ctx=ctx,
                scope="resources",
                to_uri="viking://resources",
            )

        assert tree.root.uri == "viking://resources/tt_b"
        assert tree.root.temp_uri == "viking://temp/import/tt_b"
        assert tree._candidate_uri == "viking://resources/tt_b"

    @pytest.mark.asyncio
    async def test_resources_root_to_with_trailing_slash_uses_child_incremental_target(self):
        from openviking.parse.tree_builder import TreeBuilder
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        entries = {
            "viking://temp/import": [{"name": "tt_b", "isDir": True}],
            "viking://temp/import/tt_b": [
                {"name": "a.md", "isDir": False},
                {"name": "b.md", "isDir": False},
            ],
        }
        fs = self._make_fs(entries, {"viking://resources", "viking://resources/tt_b"})
        builder = TreeBuilder()
        ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

        with patch("openviking.parse.tree_builder.get_viking_fs", return_value=fs):
            tree = await builder.finalize_from_temp(
                temp_dir_path="viking://temp/import",
                ctx=ctx,
                scope="resources",
                to_uri="viking://resources/",
            )

        assert tree.root.uri == "viking://resources/tt_b"
        assert tree.root.temp_uri == "viking://temp/import/tt_b"
        assert tree._candidate_uri == "viking://resources/tt_b"

    @pytest.mark.asyncio
    async def test_resources_root_to_flattens_single_file_document(self):
        from openviking.parse.tree_builder import TreeBuilder
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        entries = {
            "viking://temp/import": [{"name": "aa", "isDir": True}],
            "viking://temp/import/aa": [{"name": "aa.py", "isDir": False}],
        }
        fs = self._make_fs(entries, {"viking://resources"})
        builder = TreeBuilder()
        ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

        with patch("openviking.parse.tree_builder.get_viking_fs", return_value=fs):
            tree = await builder.finalize_from_temp(
                temp_dir_path="viking://temp/import",
                ctx=ctx,
                scope="resources",
                to_uri="viking://resources",
            )

        assert tree.root.uri == "viking://resources/aa"
        # Single-file code document: the resource root is the file itself,
        # not a wrapper directory named after the document.
        assert tree.root.temp_uri == "viking://temp/import/aa/aa.py"
        assert tree._candidate_uri == "viking://resources/aa"

    @pytest.mark.asyncio
    async def test_exact_to_flattens_single_file_document(self):
        from openviking.parse.tree_builder import TreeBuilder
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        entries = {
            "viking://temp/import": [{"name": "build.rs", "isDir": True}],
            "viking://temp/import/build.rs": [{"name": "build.rs", "isDir": False}],
        }
        fs = self._make_fs(entries, {"viking://resources", "viking://resources/openviking-test"})
        builder = TreeBuilder()
        ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

        with patch("openviking.parse.tree_builder.get_viking_fs", return_value=fs):
            tree = await builder.finalize_from_temp(
                temp_dir_path="viking://temp/import",
                ctx=ctx,
                scope="resources",
                to_uri="viking://resources/openviking-test/build.rs",
            )

        assert tree.root.uri == "viking://resources/openviking-test/build.rs"
        assert tree.root.temp_uri == "viking://temp/import/build.rs/build.rs"
        assert tree._candidate_uri is None

    @pytest.mark.asyncio
    async def test_single_file_non_code_document_keeps_directory(self):
        from openviking.parse.tree_builder import TreeBuilder
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        entries = {
            "viking://temp/import": [{"name": "SECURITY.md", "isDir": True}],
            "viking://temp/import/SECURITY.md": [{"name": "SECURITY.md", "isDir": False}],
        }
        fs = self._make_fs(entries, {"viking://resources", "viking://resources/openviking-test"})
        builder = TreeBuilder()
        ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

        with patch("openviking.parse.tree_builder.get_viking_fs", return_value=fs):
            tree = await builder.finalize_from_temp(
                temp_dir_path="viking://temp/import",
                ctx=ctx,
                scope="resources",
                to_uri="viking://resources/openviking-test/SECURITY.md",
            )

        # Non-code suffixes (.md is a documentation extension) keep the
        # directory form even when the parse output is a single file.
        assert tree.root.uri == "viking://resources/openviking-test/SECURITY.md"
        assert tree.root.temp_uri == "viking://temp/import/SECURITY.md"
        assert tree._candidate_uri is None

    @pytest.mark.asyncio
    async def test_document_with_artifacts_keeps_directory(self):
        from openviking.parse.tree_builder import TreeBuilder
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        entries = {
            "viking://temp/import": [{"name": "report", "isDir": True}],
            "viking://temp/import/report": [
                {"name": "report.md", "isDir": False},
                {"name": "media", "isDir": True},
            ],
        }
        fs = self._make_fs(entries, {"viking://resources"})
        builder = TreeBuilder()
        ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

        with patch("openviking.parse.tree_builder.get_viking_fs", return_value=fs):
            tree = await builder.finalize_from_temp(
                temp_dir_path="viking://temp/import",
                ctx=ctx,
                scope="resources",
                to_uri="viking://resources",
            )

        assert tree.root.uri == "viking://resources/report"
        assert tree.root.temp_uri == "viking://temp/import/report"
