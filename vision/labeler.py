"""In-notebook labelling tool for the OOPS! candidate clips (works in Google Colab).

    from labeler import Labeler
    Labeler("/content/drive/MyDrive/fall-cvpr/oops").show()

Clips from the unbiased random sample come first (they form the test set and must all be
labelled); the high-scoring candidates follow. Every click is written to labels.csv right
away, so you can stop and resume any time.

Definition used (WHO): a fall is "an event which results in a person coming to rest
inadvertently on the ground or floor or other lower level". Count it only if a person
(not an animal or object) falls, and the fall happens inside the clip.
"""
import base64, csv, os, random

LABELS = {"fall": "Fall  [F]", "not_fall": "Not a fall  [N]", "unsure": "Unsure  [U]"}
KEYS = {"f": "fall", "n": "not_fall", "u": "unsure"}

# Keyboard shortcuts in Colab: JS in the output frame calls back into Python. Only active
# after you click inside the labeller's output area (the browser sends keys to that frame).
_KEY_JS = """
if (!window.__fallKeys) {
  window.__fallKeys = true;
  document.addEventListener('keydown', (e) => {
    if (e.repeat || (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName))) return;
    const k = (e.key || '').toLowerCase();
    if (['f', 'n', 'u', 'z'].includes(k)) {
      e.preventDefault();
      google.colab.kernel.invokeFunction('fallLabeler.key', [k], {});
    }
  });
}
"""


class Labeler:
    def __init__(self, oops_dir, labeler_id="me", start_with="test", labels_file="labels.csv",
                 second_annotator=False, adjudicate=False):
        """start_with="test" (default) shows the test-set clips first; "train" skips straight
        to the training-pool clips (test clips come after).

        second_annotator=True: an independent second person labels exactly the test clips the
        first annotator labelled, into a separate file (default labels_friend.csv), without
        seeing the first annotator's answers. The first annotator's labels.csv is never touched."""
        self.dir = oops_dir
        if adjudicate:        # both annotators together settle only the clips they disagreed on
            labels_file, labeler_id = "labels_adjudicated.csv", "adjudicated"
        if second_annotator and labels_file == "labels.csv":
            labels_file, labeler_id = "labels_friend.csv", "second"
        self.path = os.path.join(oops_dir, labels_file)
        self.labeler_id = labeler_id
        with open(os.path.join(oops_dir, "scores.csv")) as fh:
            rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
        clip_dir = os.path.join(oops_dir, "clips")
        rows = [r for r in rows if os.path.exists(os.path.join(clip_dir, r["name"].replace("/", "_") + ".mp4"))]
        rnd = [r for r in rows if r["random_sample"] == "1"]
        top = [r for r in rows if r["random_sample"] == "0"]
        random.Random(0).shuffle(rnd)
        random.Random(1).shuffle(top)   # shuffled so labels are not ordered by model score
        self.queue = top + rnd if start_with == "train" else rnd + top
        if second_annotator:
            first = load_labels(oops_dir)          # only used to know WHICH clips to show
            self.queue = [r for r in rnd if first.get(r["name"]) in ("fall", "not_fall", "unsure")]
        if adjudicate:
            first = load_labels(oops_dir)
            second = {}
            with open(os.path.join(oops_dir, "labels_friend.csv")) as fh:
                for r in csv.DictReader(fh):
                    second[r["name"]] = r["label"]
            keep = ("fall", "not_fall")
            self.queue = [r for r in rnd if first.get(r["name"]) in keep and second.get(r["name"]) in keep
                          and first[r["name"]] != second[r["name"]]]
        self.n_random = len(self.queue) if (second_annotator or adjudicate) else len(rnd)
        self.test_names = {r["name"] for r in rnd}
        self.labels = {}
        if os.path.exists(self.path):
            with open(self.path) as fh:
                for r in csv.DictReader(fh):
                    self.labels[r["name"]] = r["label"]
        self.history = []

    def _save(self, name, label):
        self.labels[name] = label
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["name", "label", "labeler"])
            w.writerow([name, label, self.labeler_id])   # later rows override earlier ones

    def _next(self):
        for r in self.queue:
            if r["name"] not in self.labels:
                return r
        return None

    def show(self):
        import ipywidgets as W
        from IPython.display import display
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except Exception:
            pass
        self.video = W.HTML()
        self.info = W.HTML()
        buttons = []
        for key, text in LABELS.items():
            b = W.Button(description=text, button_style={"fall": "danger", "not_fall": "success"}.get(key, ""),
                         layout=W.Layout(width="170px", height="48px"))
            b.on_click(lambda _b, k=key: self._label(k))
            buttons.append(b)
        back = W.Button(description="← Undo  [Z]", layout=W.Layout(width="120px", height="48px"))
        back.on_click(lambda _b: self._undo())
        display(W.VBox([self.info, self.video, W.HBox(buttons + [back])]))
        try:                                   # keyboard shortcuts (Colab only)
            from google.colab import output
            from IPython.display import Javascript
            output.register_callback("fallLabeler.key", self._key)
            display(Javascript(_KEY_JS))
        except Exception:
            pass
        self._render()

    def _key(self, k):
        if k == "z":
            self._undo()
        elif k in KEYS:
            self._label(KEYS[k])

    def _render(self):
        r = self._next()
        done_rand = sum(1 for q in self.queue if q["name"] in self.test_names and q["name"] in self.labels)
        done_all = sum(1 for q in self.queue if q["name"] in self.labels)
        status = (f"<b>Test set (random sample):</b> {done_rand}/{self.n_random} &nbsp;·&nbsp; "
                  f"<b>all clips:</b> {done_all}/{len(self.queue)}")
        if r is None:
            self.info.value = status + "<br><b>All clips labelled. 🎉</b>"
            self.video.value = ""
            return
        self.current = r
        part = "TEST SET clip" if r["random_sample"] == "1" else "training-pool clip"
        self.info.value = (status + f"<br><small>{part} · the accident happens around the middle (2 s).<br>"
                           "Fall = a <b>person</b> unintentionally ends up on the ground/floor/lower level.<br>"
                           "Keyboard: click once inside this area, then press <b>F</b> / <b>N</b> / <b>U</b>, "
                           "<b>Z</b> to undo.</small>")
        with open(os.path.join(self.dir, "clips", r["name"].replace("/", "_") + ".mp4"), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        self.video.value = (f'<video src="data:video/mp4;base64,{b64}" autoplay loop muted controls '
                            f'playsinline style="max-width:480px;max-height:360px;background:#000"></video>')

    def _label(self, key):
        if getattr(self, "current", None) is None:
            return
        self._save(self.current["name"], key)
        self.history.append(self.current["name"])
        self._render()

    def _undo(self):
        if not self.history:
            return
        name = self.history.pop()
        self.labels.pop(name, None)
        self._save(name, "")          # empty label = removed
        self.labels.pop(name, None)
        self._render()


def load_labels(oops_dir):
    """name -> label, last write wins; empty labels (undone) dropped."""
    out = {}
    path = os.path.join(oops_dir, "labels.csv")
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                out[r["name"]] = r["label"]
    return {k: v for k, v in out.items() if v}
