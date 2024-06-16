from . import Agent
from . import default
import argparse


def main() -> None:
    # Setup the argument parser.
    parser = argparse.ArgumentParser('python3 -m licenceplate')
    parser.add_argument('--server', metavar='SERVER_ADDRESS', type=str, required=True, help='Endereço do servidor web para o qual o agente se reportará.')
    parser.add_argument('--video', metavar='VIDEO_URI', type=str, required=True, help='Vídeo a ser processado pelo agente. Pode ser um endereço RTSP ou um arquivo em disco.')
    parser.add_argument('--name', metavar='TAG_SLUG', type=str, required=True, help='Apelido da câmera monitorada pelo agente.')
    parser.add_argument('--output', metavar='FILENAME', type=str, required=False, default=default.OUTPUT_FILENAME, help=f'Caminho para o arquivo onde será armazenado o vídeo que ilustra o procssamento (default = {default.OUTPUT_FILENAME}).')
    parser.add_argument('--output-fourcc', metavar='FOURCC', type=str, required=False, default=default.OUTPUT_FOURCC, help=f'Código FourCC do codec a ser utilizado no vídeo de saída (default = {default.OUTPUT_FOURCC}).')
    parser.add_argument('--patience', metavar='SECONDS', type=float, required=False, default=default.TRACKING_PATIENCE_IN_SECONDS, help=f'Tempo, em segundos, para que um veículo que deixou de ser detectado seja reportado (default = {default.TRACKING_PATIENCE_IN_SECONDS}).')
    parser.add_argument('--storage-workers', metavar='NUM_THREADS', required=False, type=int, choices=range(1, 9), default=default.NUM_LOCAL_STORAGE_WORKERS, help=f'Quantidade de threads responsáveis pelo armazenamento das placas detectadas (default = {default.NUM_LOCAL_STORAGE_WORKERS}).')
    parser.add_argument('--storage-queue-size', metavar='SIZE', required=False, type=int, default=default.STORAGE_QUEUE_SIZE, help=f'Tamanho do buffer que mantém os resultados de detecção encaminhados para armazenamento (default = {default.STORAGE_QUEUE_SIZE}).')
    # Parse arguments.
    kwargs = vars(parser.parse_args())
    # Create and run the agent.
    agent = Agent(kwargs['name'])
    agent.run(
        server_address=kwargs['server'],
        video_uri=kwargs['video'],
        output_filename=kwargs['output'],
        output_fourcc=kwargs['output_fourcc'],
        tracking_patience_in_seconds=kwargs['patience'],
        num_local_storage_workers=kwargs['storage_workers'],
        storage_queue_size=kwargs['storage_queue_size'],
    )


if __name__ == '__main__':
    main()
