import os

img_rows, img_cols, img_size = 224, 224, 224
channel = 3
batch_size = 256
epochs = 10000
patience = 50
num_train_samples = 14883151
num_valid_samples = 2102270
embedding_size = 128
vocab_size = 17628
max_token_length = 40
num_image_features = 2048
hidden_size = 512

train_folder = 'data/ai_challenger_caption_train_20170902'
valid_folder = 'data/ai_challenger_caption_validation_20170910'
test_a_folder = 'data/ai_challenger_caption_train_20170902/images'
test_b_folder = 'data/ai_challenger_caption_test_b_20180103'
train_image_folder = 'data/ai_challenger_caption_train_20170902/caption_train_images_20170902'
valid_image_folder = os.path.join(valid_folder, 'caption_validation_images_20170910')
test_a_image_folder = test_a_folder
test_b_image_folder = os.path.join(test_b_folder, 'caption_test_b_images_20180103')
train_annotations_filename = 'caption_train_annotations_20170902.json'
valid_annotations_filename = 'caption_validation_annotations_20170910.json'
test_a_annotations_filename = 'caption_test_a_annotations_20180103.json'
test_b_annotations_filename = 'caption_test_b_annotations_20180103.json'

generate_image_filename = 'data/'

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<UNK>'

best_model = 'model.04-1.3820.hdf5'
beam_size = 20


class Config(object):
    num_layers = 3  # LSTM层数
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = False
    epoch = 50
    batch_size = 16
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 200  # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env = 'poetry'  # visdom env
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = "/home/marvin/PycharmProjects/QT-Create-poem/models/tang_new.pth"  # 预训练模型路径
    prefix_words = '仙路尽头谁为峰？一见无始道成空。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/tang'  # 模型保存路径
    embedding_dim = 256
    hidden_dim = 512