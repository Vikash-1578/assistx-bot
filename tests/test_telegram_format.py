import re
import unittest

from app.utils.telegram_format import md_to_html, split_markdown, to_telegram_chunks


class HtmlTests(unittest.TestCase):
    def test_escapes_raw_html_and_angle_brackets(self):
        out = md_to_html("if a < b && c > d <script>x</script>")
        self.assertNotIn("<script>", out)
        self.assertIn("a &lt; b &amp;&amp; c &gt; d", out)

    def test_bold_heading_bullets_link(self):
        out = md_to_html("# Title\n**hi** there\n- one\n* two\n[site](https://a.com/x?y=1&z=2)")
        self.assertIn("<b>Title</b>", out)
        self.assertIn("<b>hi</b> there", out)
        self.assertIn("• one\n• two", out)
        self.assertIn('<a href="https://a.com/x?y=1&amp;z=2">site</a>', out)

    def test_code_block_is_escaped_and_not_formatted(self):
        out = md_to_html("Run:\n```python\nif a<b: print('**x**')\n```\ndone")
        self.assertIn('<pre><code class="language-python">if a&lt;b: print(\'**x**\')</code></pre>', out)
        self.assertNotIn("<b>x</b>", out)

    def test_unclosed_fence_still_valid(self):
        out = md_to_html("```js\nlet a = 1")
        self.assertTrue(out.startswith("<pre><code") and out.endswith("</code></pre>"))

    def test_inline_code(self):
        self.assertEqual(md_to_html("use `a<b` here"), "use <code>a&lt;b</code> here")

    def test_unmatched_stars_left_alone(self):
        self.assertEqual(md_to_html("2 * 3 = 6 and **oops"), "2 * 3 = 6 and **oops")

    def test_tags_are_balanced(self):
        out = md_to_html("**a** `b` ```x\nc\n``` [l](http://u)")
        for tag in ("b", "code", "pre", "a"):
            self.assertEqual(out.count(f"<{tag}"), out.count(f"</{tag}>"), tag)


class SplitTests(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(split_markdown("hello"), ["hello"])
        self.assertEqual(split_markdown("  "), [])

    def test_chunks_within_limit(self):
        text = "\n".join(f"line number {i} " + "x" * 50 for i in range(300))
        chunks = split_markdown(text, 500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 520 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_code_fence_balanced_in_each_chunk(self):
        code = "\n".join(f"x{i} = {i}" for i in range(200))
        chunks = split_markdown(f"intro\n```python\n{code}\n```\noutro", 400)
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertEqual(len(re.findall(r"^```", c, re.M)) % 2, 0, c[:80])
        self.assertTrue(chunks[1].startswith("```python"))

    def test_giant_single_line(self):
        chunks = split_markdown("a" * 2500, 1000)
        self.assertEqual(sum(len(c) for c in chunks), 2500)

    def test_pairs(self):
        pairs = to_telegram_chunks("**x**")
        self.assertEqual(pairs, [("<b>x</b>", "**x**")])


if __name__ == "__main__":
    unittest.main()
