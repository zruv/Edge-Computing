import unittest
import sys
import os
from unittest.mock import MagicMock

# MOCK tflite_runtime BEFORE importing the server
# This allows tests to run on machines (like CI servers) that might not have the Edge TPU libraries installed
sys.modules['tflite_runtime'] = MagicMock()
sys.modules['tflite_runtime.interpreter'] = MagicMock()

# Now we can safely import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_server import app, load_labels

class TestEdgeAI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_labels_exist(self):
        """Check if labels file exists and has content"""
        labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coco_labels.txt')
        self.assertTrue(os.path.exists(labels_path), "coco_labels.txt not found")
        
        labels = load_labels(labels_path)
        self.assertGreater(len(labels), 0, "Labels file is empty")
        self.assertEqual(labels[0], 'person', "First label should be 'person'")

    def test_index_route(self):
        """Check if the web dashboard loads (HTTP 200)"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'EDGE VISION', response.data)

if __name__ == '__main__':
    unittest.main()
