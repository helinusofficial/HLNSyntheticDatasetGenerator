import numpy as np
from scipy.stats import truncnorm

class Normalization:
    @staticmethod
    def normalize_arrays(arr1, arr2):
        def normalize(arr):
            return (arr - arr.min()) / (arr.max() - arr.min())
        return normalize(arr1), normalize(arr2)

    @staticmethod
    def normalize_array(arr1):
        def normalize(arr):
            return (arr - arr.min()) / (arr.max() - arr.min())
        return normalize(arr1)

class DataTools:
    @staticmethod
    def generate_histo_nclip(num_samples, seed, low, high, loc, scale):
        rng = np.random.default_rng(seed)
        raw_data= rng.normal(loc=loc, scale=scale, size=num_samples)
        raw_data=np.clip(raw_data, low, high)
        return raw_data

    @staticmethod
    def generate_histo_data(num_samples, seed,low,high,loc,scale):
            rng = np.random.default_rng(seed)
            a, b = (low - loc) / scale, (high - loc) / scale
            raw_data = truncnorm.rvs(a, b, loc=loc, scale=scale, size=num_samples, random_state=rng)
            return raw_data

    @staticmethod
    def generate_discrete_histo_data(num_samples, seed, low, high, loc, scale):
        rng = np.random.default_rng(seed)
        values = np.arange(low, high+1)

        # محاسبه احتمال هر مقدار صحیح بر اساس pdf نرمال بریده
        a, b = (low - loc) / scale, (high - loc) / scale
        probs = truncnorm.pdf(values, a, b, loc=loc, scale=scale)
        probs = probs / probs.sum()  # نرمال‌سازی به مجموع ۱

        # نمونه‌گیری گسسته
        discrete_data = rng.choice(values, size=num_samples, p=probs)

        return discrete_data

    @staticmethod
    def generate_rejection(num_samples, seed, low, high, loc, scale):
        rng = np.random.default_rng(seed)
        samples = []
        while len(samples) < num_samples:
            x = rng.normal(loc=loc, scale=scale)
            if low <= x <= high:
                samples.append(x)
        return np.array(samples)

    @staticmethod
    def generate_discrete_histo_data_age(num_samples, seed, low, high, loc, scale):
        rng = np.random.default_rng(seed)
        a, b = (low - loc) / scale, (high - loc) / scale
        ages = truncnorm.rvs(a, b, loc=loc, scale=scale, size=num_samples, random_state=rng)
        years = np.floor(ages).astype(int)
        months = np.round((ages - years) * 12).astype(int)
        months = np.where(months == 12, 0, months)
        years += (months == 0) & (np.round((ages - np.floor(ages)) * 12) == 12)
        return years + months / 12.0
