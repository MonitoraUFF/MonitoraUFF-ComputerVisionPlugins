def on_open(agent, data):
    agent.send({
        'type': 'agent-report',
        'event': 'open',
        'who': agent.tag_slug,
        'data': {},
    })


def on_frame(agent, data):
    pass


def on_plate(agent, data):
    pass
    

def on_close(agent, data):
    agent.send({
        'type': 'agent-report',
        'event': 'close',
        'who': agent.tag_slug,
        'data': {},
    })


def on_error(agent, data):
    pass


CALLBACKS = {
    'open': on_open,
    'frame': on_frame,
    'plate': on_plate,
    'close': on_close,
    'error': on_error,
}