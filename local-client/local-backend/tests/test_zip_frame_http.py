import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import main


class ZipFrameHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_includes_waiting_for_decode_slot(self):
        with (
            patch.object(main, "_zip_frame_semaphore", asyncio.Semaphore(0)),
            patch.object(main, "ZIP_FRAME_TIMEOUT_SEC", 0.01, create=True),
        ):
            with self.assertRaises(HTTPException) as raised:
                await asyncio.wait_for(main.zip_frame("L26_V406", 87320), timeout=0.2)

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(raised.exception.detail, "Frame request timed out for L26_V406")


if __name__ == "__main__":
    unittest.main()
