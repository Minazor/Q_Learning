import gym
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# Deprecation uyarılarını görmezden gel
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Öğrenme parametreleri
LEARNING_RATE = 0.1  # Q-learning için öğrenme oranı
DISCOUNT = 0.95  # Gelecek ödüller için indirim faktörü
EPISODES = 25000  # Toplam eğitim bölümü sayısı

SHOW_EVERY = 2000  # Kaç bölümde bir sonuçların gösterileceği

# MountainCar-v0 ortamını oluştur
env = gym.make("MountainCar-v0")

# Gözlem uzayını ayrıklaştırma boyutu (her gözlem boyutu için 20 aralık)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# Ayrık gözlem uzayı penceresi boyutu (gözlem uzayını belirli sayıda aralığa bölmek için)
discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE

epsilon = 0.5  # Epsilon-greedy politika için başlangıç epsilon değeri
START_EPSILON_DECAYING = 1  # Epsilon'un azalmaya başlayacağı bölüm
END_EPSILON_DECAYING = EPISODES // 2  # Epsilon'un azalmayı durduracağı bölüm

# Epsilon'un azalacağı değer
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Q-tablosunu rastgele değerlerle başlat (20x20x3 boyutlarında)
# Her durum-eylem çifti için başlangıç Q değeri -2 ile 0 arasında rastgele seçilir
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)

ep_rewards = []  # Her bölüm için ödülleri saklamak için liste
aggr_ep_reward = {
    "ep": [],
    "avg": [],
    "min": [],
    "max": [],
}  # Her SHOW_EVERY bölümde bir ortalama, minimum ve maksimum ödülleri saklamak için sözlük

# Q-tablolarını saklamak için klasör oluştur
if not os.path.exists("qtables"):
    os.makedirs("qtables")


# Sürekli durumu ayrık duruma çeviren fonksiyon
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(
        discrete_state.astype(int)
    )  # Ayrık durumu tamsayıya çevirip tuple olarak döndür


# Ana eğitim döngüsü
for episode in range(EPISODES):
    episode_reward = 0  # Her bölüm için ödülü sıfırla
    render = False  # Başlangıçta görselleştirmeyi kapalı tut

    # SHOW_EVERY bölümde bir görselleştirme yap
    if episode % SHOW_EVERY == 0:
        render = True
        print(f"Episode: {episode}")  # Bölüm numarasını yazdır

    # Ortamı sıfırla ve ilk durumu al
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)
    observation, info = env.reset()
    discrete_state = get_discrete_state(
        observation
    )  # Sürekli durumu ayrık duruma çevir

    done = False  # Bölüm bitiş durumu
    while not done:
        # Epsilon-greedy politika ile eylem seç
        if np.random.random() > epsilon:
            action = np.argmax(
                q_table[discrete_state]
            )  # Q-tablosuna göre en iyi eylemi seç
        else:
            action = np.random.randint(0, env.action_space.n)  # Rastgele bir eylem seç

        # Eylemi gerçekleştir ve yeni durumu ve ödülü gözlemle
        new_state, reward, termination, truncation, _ = env.step(action)
        episode_reward += reward  # Bölüm ödülüne ekle
        new_discrete_state = get_discrete_state(
            new_state
        )  # Yeni durumu ayrık duruma çevir

        # Eğer bölüm sona ermişse
        if termination or truncation:
            done = True  # Bölüm sona erdi
        if not done:
            # Gelecekteki maksimum Q-değerini hesapla
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            # Q-değerini güncelle
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            # Hedefe ulaşıldığında Q-değerini sıfırla
            print(f"Reach the goal on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        # Gelecek durumu güncelle
        discrete_state = new_discrete_state

    # Epsilon azaltma
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Bölüm ödülünü sakla
    ep_rewards.append(episode_reward)

    # Her 10 bölümde bir Q-tablosunu kaydet
    if not episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

    # SHOW_EVERY bölümde bir ödül istatistiklerini hesapla ve göster
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_reward["ep"].append(episode)
        aggr_ep_reward["avg"].append(average_reward)
        aggr_ep_reward["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_reward["max"].append(max(ep_rewards[-SHOW_EVERY:]))

        print(
            f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}"
        )

# Ortamı kapat
env.close()

# Ödül istatistiklerini çizdir
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["avg"], label="avg")
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["min"], label="min")
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["max"], label="max")
plt.legend(loc=4)
plt.show()
