"""Helper functions for processing the graph-structured OpenAssistant dataset."""
from typing import Callable, Dict, List, Set
import tokenizer.profiles as tokenizer_profiles

def build_graph_structures(dataset_split):
    """Builds graph data structures from a dataset split."""
    message_id_to_message = {}
    parent_id_to_message_ids = {}
    root_ids = []
    for message in dataset_split:
        message_id = message['message_id']
        parent_id = message['parent_id']
        message_id_to_message[message_id] = message
        if parent_id:
            if parent_id not in parent_id_to_message_ids:
                parent_id_to_message_ids[parent_id] = []
            parent_id_to_message_ids[parent_id].append(message_id)
        else:
            root_ids.append(message_id)
    return message_id_to_message, parent_id_to_message_ids, root_ids

def get_thread_messages(root_id: str, parent_id_to_message_ids: Dict[str, List[str]]) -> Set[str]:
    """Collects all message_ids belonging to a single conversation thread."""
    thread_messages = set()
    q = [root_id]
    thread_messages.add(root_id)
    while q:
        current_id = q.pop(0)
        child_ids = parent_id_to_message_ids.get(current_id, [])
        q.extend(child_ids)
        thread_messages.update(child_ids)
    return thread_messages

def traverse_thread(
    root_id: str, 
    message_id_to_message: Dict[str, Dict], 
    parent_id_to_message_ids: Dict[str, List[str]], 
    tokenizer_profile: tokenizer_profiles.TokenizerProfile):
    """Traverses a single conversation thread and yields training examples."""
    q = [(root_id, [message_id_to_message[root_id]])]

    while q:
        current_id, context = q.pop(0)
        
        child_ids = parent_id_to_message_ids.get(current_id, [])
        
        if not child_ids:
            continue

        if len(child_ids) == 1:
            # Continue conversation path
            new_context = context + [message_id_to_message[child_ids[0]]]
            q.append((child_ids[0], new_context))
        else:
            # Branching point, create a training example
            sorted_responses = sorted([message_id_to_message[cid] for cid in child_ids], key=lambda x: x.get('rank', 0) or 0)
            for r1_idx in range(len(sorted_responses)):
                for r2_idx in range(r1_idx + 1, len(sorted_responses)):
                    r1 = sorted_responses[r1_idx]
                    r2 = sorted_responses[r2_idx]
                    # context: list[message], responses. tuple(message, message).
                    yield tokenizer_profile.format_example({'context': context, 'accepted': r1, 'rejected': r2}, mode='openassistant')
            # Continue traversal with the highest-ranked response
            q.append((sorted_responses[0]['message_id'], context + [sorted_responses[0]]))
