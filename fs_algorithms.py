"""
fs_algorithms.py
خوارزميات السرب لاختيار الخصائص (Wrapper Feature Selection)

كل "حل" = متجه أرقام بين 0 و 1 بطول عدد الخصائص.
الخاصية تُختار إذا كانت قيمتها > 0.5  (تحويل بسيط من مستمر إلى ثنائي).

الخوارزميات:
    1) PSO     - Particle Swarm Optimization (الأساس في الأوراق السابقة)
    2) GWO     - Grey Wolf Optimizer
    3) PSOGWO  - الهجينة (الطريقة المستخدمة في الورقة الأصلية)
    4) LGWO    - GWO مع بحث محلي بقفزة Lévy (Heidari & Pahlavani 2017)

كل الخوارزميات تستخدم نفس عدد الحلول (20) ونفس عدد التكرارات (50)
=> 1,020 طلب لدالة اللياقة، ما عدا LGWO: 1,070 (طلب زيادة في كل تكرار).
"""
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# ------------------------------------------------------------------
# دالة اللياقة (Fitness): كلما كانت أصغر كان الحل أفضل
#   fitness = alpha * الخطأ + (1 - alpha) * (عدد الخصائص المختارة / الكل)
# ------------------------------------------------------------------
class Fitness:
    def __init__(self, X, y, alpha=0.99, seed=0, clf=None, scoring="accuracy"):
        self.X, self.y, self.alpha = X, y, alpha
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        self.clf = clf or KNeighborsClassifier(n_neighbors=5)   # KNN سريع => مناسب داخل الـ wrapper
        self.scoring = scoring   # "balanced_accuracy" للبيانات غير المتوازنة
        self.cache = {}                                  # لتجنب إعادة حساب نفس المجموعة
        self.calls = 0                                   # كم مرة انطلبت دالة اللياقة (الميزانية)
        self.evals = 0                                   # كم مرة تدرب المصنف فعلاً (بدون الـ cache)

    def __call__(self, position):
        self.calls += 1
        mask = position > 0.5
        if not mask.any():                 # لازم خاصية واحدة على الأقل
            return 1.0
        key = mask.tobytes()
        if key not in self.cache:
            self.evals += 1
            acc = cross_val_score(self.clf, self.X[:, mask], self.y, cv=self.cv, scoring=self.scoring).mean()
            ratio = mask.sum() / mask.size
            self.cache[key] = self.alpha * (1 - acc) + (1 - self.alpha) * ratio
        return self.cache[key]


def _init(n_agents, dim, rng):
    return rng.random((n_agents, dim))


# ------------------------------------------------------------------
# 1) PSO
# ------------------------------------------------------------------
def pso(fit, dim, n_agents=20, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    X = _init(n_agents, dim, rng)
    V = np.zeros_like(X)
    pbest, pbest_f = X.copy(), np.array([fit(x) for x in X])
    g = pbest_f.argmin()
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    curve = []
    for t in range(n_iter):
        w = 0.9 - 0.5 * t / n_iter                      # الوزن يتناقص من 0.9 إلى 0.4
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + 2 * r1 * (pbest - X) + 2 * r2 * (gbest - X)
        V = np.clip(V, -0.5, 0.5)
        X = np.clip(X + V, 0, 1)
        for i in range(n_agents):
            f = fit(X[i])
            if f < pbest_f[i]:
                pbest[i], pbest_f[i] = X[i].copy(), f
                if f < gbest_f:
                    gbest, gbest_f = X[i].copy(), f
        curve.append(gbest_f)
    return gbest > 0.5, gbest_f, curve


# ------------------------------------------------------------------
# أداة مشتركة: ترتيب الذئاب (alpha, beta, delta = أفضل 3 حلول)
# ------------------------------------------------------------------
def _update_leaders(X, F, leaders, leaders_f):
    for i in range(len(X)):
        f = F[i]
        if f < leaders_f[0]:
            leaders[2], leaders_f[2] = leaders[1].copy(), leaders_f[1]
            leaders[1], leaders_f[1] = leaders[0].copy(), leaders_f[0]
            leaders[0], leaders_f[0] = X[i].copy(), f
        elif f < leaders_f[1]:
            leaders[2], leaders_f[2] = leaders[1].copy(), leaders_f[1]
            leaders[1], leaders_f[1] = X[i].copy(), f
        elif f < leaders_f[2]:
            leaders[2], leaders_f[2] = X[i].copy(), f


def _gwo_move(X, leaders, a, rng):
    """حركة الذئاب: كل ذئب يتحرك نحو متوسط مواقع alpha و beta و delta"""
    new = np.zeros_like(X)
    for L in leaders:
        A = 2 * a * rng.random(X.shape) - a
        C = 2 * rng.random(X.shape)
        D = np.abs(C * L - X)
        new += L - A * D
    return new / 3.0


# ------------------------------------------------------------------
# 2) GWO
# ------------------------------------------------------------------
def gwo(fit, dim, n_agents=20, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    X = _init(n_agents, dim, rng)
    F = np.array([fit(x) for x in X])
    leaders = [X[0].copy() for _ in range(3)]
    leaders_f = [np.inf] * 3
    _update_leaders(X, F, leaders, leaders_f)
    curve = []
    for t in range(n_iter):
        a = 2 - 2 * t / n_iter                          # يتناقص من 2 إلى 0 (استكشاف ← استغلال)
        X = np.clip(_gwo_move(X, leaders, a, rng), 0, 1)
        F = np.array([fit(x) for x in X])
        _update_leaders(X, F, leaders, leaders_f)
        curve.append(leaders_f[0])
    return leaders[0] > 0.5, leaders_f[0], curve


# ------------------------------------------------------------------
# 3) PSOGWO الهجينة (Singh & Singh 2017) - طريقة الورقة الأصلية
#    سرعة PSO لكن الاتجاه نحو الذئاب الثلاثة بدل pbest/gbest
# ------------------------------------------------------------------
def psogwo(fit, dim, n_agents=20, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    X = _init(n_agents, dim, rng)
    V = 0.3 * rng.random(X.shape)
    F = np.array([fit(x) for x in X])
    leaders = [X[0].copy() for _ in range(3)]
    leaders_f = [np.inf] * 3
    _update_leaders(X, F, leaders, leaders_f)
    curve = []
    w = 0.5
    for t in range(n_iter):
        a = 2 - 2 * t / n_iter
        Xs = []
        for L in leaders:
            A = 2 * a * rng.random(X.shape) - a
            C = 2 * rng.random(X.shape)
            Xs.append(L - A * np.abs(C * L - w * X))
        r = [rng.random(X.shape) for _ in range(3)]
        V = w * (V + 0.5 * r[0] * (Xs[0] - X) + 0.5 * r[1] * (Xs[1] - X) + 0.5 * r[2] * (Xs[2] - X))
        V = np.clip(V, -0.5, 0.5)
        X = np.clip(X + V, 0, 1)
        F = np.array([fit(x) for x in X])
        _update_leaders(X, F, leaders, leaders_f)
        curve.append(leaders_f[0])
    return leaders[0] > 0.5, leaders_f[0], curve


# ------------------------------------------------------------------
# 4) LGWO: GWO + Lévy flight
#    الفكرة: بعد حركة GWO العادية، نجربوا حل واحد جديد = الذئب القائد alpha + قفزة Lévy.
#    قفزات Lévy أغلبها صغيرة وأحياناً كبيرة جداً
#    => تساعد على الهروب من الحلول المحلية (local optima).
#    حجم القفزة ثابت (scale = 0.1)، ونقبلوا الحل الجديد بس لو أحسن من alpha.
# ------------------------------------------------------------------
def levy(shape, rng, beta=1.5):
    """توليد خطوات Lévy بطريقة Mantegna"""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, shape)
    v = rng.normal(0, 1, shape)
    return u / np.abs(v) ** (1 / beta)


def lgwo(fit, dim, n_agents=20, n_iter=50, seed=0, scale=0.1):
    """
    GWO عادي + بحث محلي بقفزة Lévy حول الذئب القائد (alpha):
      في كل تكرار نولّد حلاً جديداً = alpha + قفزة Lévy
      إذا كان أفضل من alpha نعتمده (greedy) وإلا نتجاهله.
    التكلفة الإضافية: تقييم واحد فقط لكل تكرار (~5% زيادة).
    """
    rng = np.random.default_rng(seed)
    X = _init(n_agents, dim, rng)
    F = np.array([fit(x) for x in X])
    leaders = [X[0].copy() for _ in range(3)]
    leaders_f = [np.inf] * 3
    _update_leaders(X, F, leaders, leaders_f)
    curve = []
    for t in range(n_iter):
        a = 2 - 2 * t / n_iter
        X = np.clip(_gwo_move(X, leaders, a, rng), 0, 1)
        F = np.array([fit(x) for x in X])
        _update_leaders(X, F, leaders, leaders_f)
        # ---- قفزة Lévy حول alpha ----
        cand = np.clip(leaders[0] + scale * levy(dim, rng), 0, 1)
        f_cand = fit(cand)
        if f_cand < leaders_f[0]:
            _update_leaders(cand[None, :], np.array([f_cand]), leaders, leaders_f)
        curve.append(leaders_f[0])
    return leaders[0] > 0.5, leaders_f[0], curve


ALGORITHMS = {"PSO": pso, "GWO": gwo, "PSOGWO": psogwo, "LGWO": lgwo}
