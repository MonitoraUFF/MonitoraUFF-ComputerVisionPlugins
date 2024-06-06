import sys
import argparse
from . import Agent


def main(argsv=None):
    # Setup the argument parser.
    parser = argparse.ArgumentParser('python3 -m licenceplate')
    parser.add_argument('--server', metavar='SERVER_ADDRESS', required=True, help='Endereço do servidor web para o qual o agente se reportará.')
    parser.add_argument('--video', metavar='VIDEO_URI', required=True, help='Vídeo a ser processado pelo agente. Pode ser um endereço RTSP ou um arquivo em disco.')
    parser.add_argument('--name', metavar='TAG_SLUG', required=True, help='Apelido da câmera monitorada pelo agente.')
    parser.add_argument('--nframe', metavar='NUM_THREADS', required=False, type=int, choices=range(1, 9), default=2, help='Quantidade de threads responsáveis pelo processamento dos quadros do vídeo (default = 2).')
    parser.add_argument('--nstorage', metavar='NUM_THREADS', required=False, type=int, choices=range(1, 9), default=2, help='Quantidade de threads responsáveis pelo armazenamento das placas detectadas (default = 2).')
    # Parse arguments.
    if argsv is None:
        argsv = sys.argv[1:]
    args = vars(parser.parse_args(argsv))
    # Create and run the agent.
    agent = Agent(args['name'])
    agent.run(
        server_address=args['server'],
        video_uri=args['video'],
        num_frame_workers=args['nframe'],
        num_local_storage_workers=args['nstorage'],
    )


if __name__ == '__main__':
    main()
