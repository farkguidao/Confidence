import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from utils import DataHub, expand_list, iterable_support
from utils.dict import PieceAlphabet
import pytorch_lightning as pl

from utils.help import noise_augment
from utils.load import build_embedding_matrix


class Dataloader(pl.LightningDataModule):
    def __init__(self,dataset,data_path,batch_size,pretrained_model='none'):
        super(Dataloader, self).__init__()
        self.save_hyperparameters()
        data_house = DataHub.from_dir_addadj(data_path)
        # piece_vocab = PieceAlphabet("piece", pretrained_model=pretrained_model)
        #
        # self._piece_vocab = piece_vocab
        self._pretrained_model = pretrained_model

        self._word_vocab = data_house.word_vocab
        self._sent_vocab = data_house.sent_vocab
        self._act_vocab = data_house.act_vocab
        self._adj_vocab = data_house.adj_vocab
        self._adj_full_vocab = data_house.adj_full_vocab
        self._adj_id_vocab = data_house.adj_id_vocab
        self._data_house = data_house

        self.num_sent = len(self._sent_vocab)
        self.num_act = len(self._act_vocab)

        self.initial_embedding_matrix = build_embedding_matrix(
                    word2idx=self._word_vocab._elem_to_idx,
                    embed_dim= 300,
                    dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(300), dataset))

    def _collate_func(self,instance_list):
        """
        As a function parameter to instantiate the DataLoader object.
        """

        n_entity = len(instance_list[0])
        scatter_b = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(instance_list)):
            for jdx in range(0, n_entity):
                scatter_b[jdx].append(instance_list[idx][jdx])
        utt_list, sent_list, act_list, adj_list, adj_full_list, adj_id_list = scatter_b

        var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, False)



        len_list = [len(i) for i in len_list]

        # 把类别映射成id
        flat_sent = iterable_support(self._sent_vocab.index, sent_list)
        flat_act = iterable_support(self._act_vocab.index, act_list)

        index_sent = expand_list(flat_sent)
        index_act = expand_list(flat_act)

        sent_label = torch.LongTensor(index_sent)
        act_label = torch.LongTensor(index_act)

        return var_utt, var_p, len_list, mask, var_adj, sent_label,act_label
    def _collate_func_with_noise(self,instance_list):
        """
        As a function parameter to instantiate the DataLoader object.
        """

        n_entity = len(instance_list[0])
        scatter_b = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(instance_list)):
            for jdx in range(0, n_entity):
                scatter_b[jdx].append(instance_list[idx][jdx])
        utt_list, sent_list, act_list, adj_list, adj_full_list, adj_id_list = scatter_b

        var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, True)

        len_list = [len(i) for i in len_list]

        # 把类别映射成id
        flat_sent = iterable_support(self._sent_vocab.index, sent_list)
        flat_act = iterable_support(self._act_vocab.index, act_list)

        index_sent = expand_list(flat_sent)
        index_act = expand_list(flat_act)

        sent_label = torch.LongTensor(index_sent)
        act_label = torch.LongTensor(index_act)

        return var_utt, var_p, len_list, mask, var_adj, sent_label,act_label
    @property
    def sent_vocab(self):
        return self._sent_vocab

    @property
    def act_vocab(self):
        return self._act_vocab

    def _wrap_padding(self, dial_list, adj_list, adj_full_list, adj_id_list, use_noise):
        dial_len_list = [len(d) for d in dial_list]
        max_dial_len = max(dial_len_list)

        adj_len_list = [len(adj) for adj in adj_list]
        max_adj_len = max(adj_len_list)

        # add adj_full
        adj_full_len_list = [len(adj_full) for adj_full in adj_full_list]
        max_adj_full_len = max(adj_full_len_list)

        # add adj_I
        adj_id_len_list = [len(adj_I) for adj_I in adj_id_list]
        max_adj_id_len = max(adj_id_len_list)

        assert max_dial_len == max_adj_len, str(max_dial_len) + " " + str(max_adj_len)
        assert max_adj_full_len == max_adj_len, str(max_adj_full_len) + " " + str(max_adj_len)
        assert max_adj_id_len == max_adj_full_len, str(max_adj_id_len) + " " + str(max_adj_full_len)

        turn_len_list = [[len(u) for u in d] for d in dial_list]
        max_turn_len = max(expand_list(turn_len_list))

        turn_adj_len_list = [[len(u) for u in adj] for adj in adj_list]
        max_turn_adj_len = max(expand_list(turn_adj_len_list))

        # add adj_full
        turn_adj_full_len_list = [[len(u) for u in adj_full] for adj_full in adj_full_list]
        max_turn_adj_full_len = max(expand_list(turn_adj_full_len_list))

        # add adj_I
        turn_adj_id_len_list = [[len(u) for u in adj_I] for adj_I in adj_id_list]
        max_turn_adj_id_len = max(expand_list(turn_adj_id_len_list))

        pad_adj_list = []
        for dial_i in range(0, len(adj_list)):
            pad_adj_list.append([])

            for turn in adj_list[dial_i]: # padding  补足后面的空余，都变成最长字符串
                pad_utt = turn + [0] * (max_turn_adj_len - len(turn))
                pad_adj_list[-1].append(pad_utt)

            if len(adj_list[dial_i]) < max_adj_len:
                pad_dial = [[0] * max_turn_adj_len] * (max_adj_len - len(adj_list[dial_i]))
                pad_adj_list[-1].extend(pad_dial)

        pad_adj_full_list = []
        for dial_i in range(0, len(adj_full_list)):
            pad_adj_full_list.append([])

            for turn in adj_full_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_full_len - len(turn))
                pad_adj_full_list[-1].append(pad_utt)

            if len(adj_full_list[dial_i]) < max_adj_full_len:
                pad_dial = [[0] * max_turn_adj_full_len] * (max_adj_full_len - len(adj_full_list[dial_i]))
                pad_adj_full_list[-1].extend(pad_dial)

        pad_adj_id_list = []
        for dial_i in range(0, len(adj_id_list)):
            pad_adj_id_list.append([])

            for turn in adj_id_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_id_len - len(turn))
                pad_adj_id_list[-1].append(pad_utt)

            if len(adj_id_list[dial_i]) < max_adj_id_len:
                pad_dial = [[0] * max_turn_adj_id_len] * (max_adj_id_len - len(adj_id_list[dial_i]))
                pad_adj_id_list[-1].extend(pad_dial)

        pad_adj_R_list = []
        for dial_i in range(0, len(pad_adj_id_list)):
            pad_adj_R_list.append([])
            assert len(pad_adj_id_list[dial_i]) == len(pad_adj_full_list[dial_i])
            for i in range(len(pad_adj_full_list[dial_i])): # 从n变成2n
                full = pad_adj_full_list[dial_i][i]
                pad_utt_up = full + full
                pad_adj_R_list[-1].append(pad_utt_up)

            for i in range(len(pad_adj_full_list[dial_i])):
                full = pad_adj_full_list[dial_i][i]
                pad_utt_down = full + full
                pad_adj_R_list[-1].append(pad_utt_down)

        assert len(pad_adj_id_list[0]) * 2 == len(pad_adj_R_list[0]), pad_adj_R_list[0]

        pad_w_list, pad_sign = [], self._word_vocab.PAD_SIGN
        for dial_i in range(0, len(dial_list)):
            pad_w_list.append([])

            for turn in dial_list[dial_i]:
                if use_noise:
                    noise_turn = noise_augment(self._word_vocab, turn, 5.0)
                else:
                    noise_turn = turn
                pad_utt = noise_turn + [pad_sign] * (max_turn_len - len(turn))
                # a = iterable_support(self._word_vocab.index, pad_utt)
                # print(a)
                pad_w_list[-1].append(iterable_support(self._word_vocab.index, pad_utt)) # 话语map成id

            if len(dial_list[dial_i]) < max_dial_len:
                pad_dial = [[pad_sign] * max_turn_len] * (max_dial_len - len(dial_list[dial_i]))
                pad_w_list[-1].extend(iterable_support(self._word_vocab.index, pad_dial))

        # cls_sign = self._piece_vocab.CLS_SIGN
        # piece_list, sep_sign = [], self._piece_vocab.SEP_SIGN
        #
        # for dial_i in range(0, len(dial_list)): # 给每句话的开头结尾加上CLS
        #     piece_list.append([])
        #
        #     for turn in dial_list[dial_i]:
        #         seg_list = self._piece_vocab.tokenize(turn)
        #         piece_list[-1].append([cls_sign] + seg_list + [sep_sign])
        #
        #     if len(dial_list[dial_i]) < max_dial_len:
        #         pad_dial = [[cls_sign, sep_sign]] * (max_dial_len - len(dial_list[dial_i]))
        #         piece_list[-1].extend(pad_dial)
        #
        # p_len_list = [[len(u) for u in d] for d in piece_list]
        # max_p_len = max(expand_list(p_len_list))
        #
        # pad_p_list, mask = [], []
        # for dial_i in range(0, len(piece_list)): # padding
        #     pad_p_list.append([])
        #     mask.append([])
        #
        #     for turn in piece_list[dial_i]:
        #         pad_t = turn + [pad_sign] * (max_p_len - len(turn))
        #         pad_p_list[-1].append(self._piece_vocab.index(pad_t))
        #         mask[-1].append([1] * len(turn) + [0] * (max_p_len - len(turn)))

        var_w_dial = torch.LongTensor(pad_w_list)
        # var_p_dial = torch.LongTensor(pad_p_list)
        # var_mask = torch.LongTensor(mask)
        var_adj_dial = torch.LongTensor(pad_adj_list)
        var_adj_full_dial = torch.LongTensor(pad_adj_full_list)
        var_adj_R_dial = torch.LongTensor(pad_adj_R_list)


        return var_w_dial, None, None, turn_len_list, None, var_adj_dial, var_adj_full_dial, \
            var_adj_R_dial

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._data_house.get_iterator("train", self.hparams.batch_size, True,collate_func=self._collate_func_with_noise)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._data_house.get_iterator("test", self.hparams.batch_size, False,collate_func=self._collate_func)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._data_house.get_iterator("dev", self.hparams.batch_size, False,collate_func=self._collate_func)


if __name__ == '__main__':
    dataloader = Dataloader('mastodon',r'../data/mastodon',16,'none')



