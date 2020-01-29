import json

from .transaction import Transaction

class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis

    def add_transaction(self, tip):
        self.transactions[tip.name()] = tip

    def show(self):
        graph = nx.DiGraph()

        graph.add_edges_from([(id(x), id(x.p1)) for x in self.transactions if x.p1 is not None])
        graph.add_edges_from([(id(x), id(x.p2)) for x in self.transactions if x.p2 is not None])

        val_map = {id(x): x.height for x in self.transactions}
        values = [val_map.get(node) for node in graph.nodes()]

        # Need to create a layout when doing
        # separate calls to draw nodes and edges
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=100)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), arrows=False)
        plt.show()

    def save(self, sequence_no, global_loss, global_accuracy, norm):
        # Mark untagged transactions with the sequence number
        for _, t in self.transactions.items():
            if t.tag is None:
                t.add_tag(sequence_no)

        n = [{'name': t.name(), 'time': t.tag, 'parents': t.parents} for _, t in self.transactions.items()]

        with open(f'tangle_data/tangle_{sequence_no}.json', 'w') as outfile:
            json.dump({'nodes': n, 'genesis': self.genesis, 'global_loss': global_loss, 'global_accuracy': global_accuracy, 'norm': norm}, outfile)

    @classmethod
    def fromfile(cls, sequence_no):
      with open(f'tangle_data/tangle_{sequence_no}.json', 'r') as tanglefile:
          t = json.load(tanglefile)

      transactions = {n['name']: Transaction(None, set(n['parents']), n['name'], n['time']) for n in t['nodes']}
      return cls(transactions, t['genesis'])
