from . import DEFAULT_VEHICLE_QUEUE_SIZE, DEFAULT_NUM_VEHICLE_WORKERS, DEFAULT_NUM_LOCAL_STORAGE_WORKERS, DEFAULT_STORAGE_QUEUE_SIZE, Agent
import argparse


def main() -> None:
    # Setup the argument parser.
    parser = argparse.ArgumentParser('python3 -m licenceplate')
    parser.add_argument('--server', metavar='SERVER_ADDRESS', required=True, help='Endereço do servidor web para o qual o agente se reportará.')
    parser.add_argument('--video', metavar='VIDEO_URI', required=True, help='Vídeo a ser processado pelo agente. Pode ser um endereço RTSP ou um arquivo em disco.')
    parser.add_argument('--name', metavar='TAG_SLUG', required=True, help='Apelido da câmera monitorada pelo agente.')
    parser.add_argument('--vehicle_workers', metavar='NUM_THREADS', required=False, type=int, choices=range(1, 9), default=DEFAULT_NUM_VEHICLE_WORKERS, help=f'Quantidade de threads responsáveis pelo processamento dos quadros do vídeo (default = {DEFAULT_NUM_VEHICLE_WORKERS}).')
    parser.add_argument('--storage_workers', metavar='NUM_THREADS', required=False, type=int, choices=range(1, 9), default=DEFAULT_NUM_LOCAL_STORAGE_WORKERS, help=f'Quantidade de threads responsáveis pelo armazenamento das placas detectadas (default = {DEFAULT_NUM_LOCAL_STORAGE_WORKERS}).')
    parser.add_argument('--vehicle_queue_size', metavar='SIZE', required=False, type=int, default=DEFAULT_VEHICLE_QUEUE_SIZE, help=f'Tamanho do buffer que mantém as imagens de veículos não processadas (default = {DEFAULT_VEHICLE_QUEUE_SIZE}).')
    parser.add_argument('--storage_queue_size', metavar='SIZE', required=False, type=int, default=DEFAULT_STORAGE_QUEUE_SIZE, help=f'Tamanho do buffer que mantém os resultados de detecção encaminhados para armazenamento (default = {DEFAULT_STORAGE_QUEUE_SIZE}).')
    # Parse arguments.
    kwargs = vars(parser.parse_args())
    # Create and run the agent.
    agent = Agent(kwargs['name'])
    agent.run(
        server_address=kwargs['server'],
        video_uri=kwargs['video'],
        num_vehicle_workers=kwargs['vehicle_workers'],
        num_local_storage_workers=kwargs['storage_workers'],
        vehicle_queue_size=kwargs['vehicle_queue_size'],
        storage_queue_size=kwargs['storage_queue_size'],
    )


if __name__ == '__main__':
    main()
