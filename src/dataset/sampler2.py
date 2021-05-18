import time
import numpy as np

from queue import Queue
import dataset.utils as utils


class SerialSampler():

    def __init__(self, data, args, sampled_classes, source_classes, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes
        self.sampled_classes = sampled_classes
        self.source_classes = source_classes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()
        self.worker(self.done_queue, self.sampled_classes, self.source_classes)


    def get_epoch(self):

        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue, sampled_classes, source_classes):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > self.num_episodes:
                time.sleep(1)
                return
                # continue

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                    self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                    self.idx_list[y][
                        tmp[self.args.shot:self.args.shot + self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                          support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                        query_idx, max_query_len)

            done_queue.put((support, query))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''

        del self.done_queue



def task_sampler(data, args):
    all_classes = np.unique(data['label'])
    num_classes = len(all_classes)

    # sample classes
    temp = np.random.permutation(num_classes)
    sampled_classes = temp[:args.way]

    source_classes = temp[args.way:args.way + args.way]

    return sampled_classes, source_classes
