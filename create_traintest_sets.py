import os
import shutil
import random

def create_traintest_sets(source_dir, destination_dir, train_percentage=0.8):
    # Remover diretório de destino antes (lixo)
    shutil.rmtree(destination_dir)
    
    # Criar diretório de destino
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Lista de subdiretórios no diretório de origem
    subdirectories = ['asc-h', 'asc-us', 'hsil', 'lsil', 'nfil', 'scc']

    for subdir in subdirectories:
        source_subdir = os.path.join(source_dir, subdir)
        destination_train_subdir = os.path.join(destination_dir, 'train', subdir)
        destination_test = os.path.join(destination_dir, 'test')

        # Criar subdiretórios de treino e teste se não existirem
        os.makedirs(destination_train_subdir, exist_ok=True)
        os.makedirs(destination_test, exist_ok=True)

        # Listar todas as imagens no subdiretório de origem
        images = os.listdir(source_subdir)

        # Calcular o número de imagens para o conjunto de treino
        num_train_images = int(len(images) * train_percentage)

        # Selecionar aleatoriamente as imagens para o conjunto de treino
        train_images = random.sample(images, num_train_images)

        # Mover imagens para os diretórios correspondentes
        for image in images:
            source_path = os.path.join(source_subdir, image)
            if image in train_images:
                destination_path = os.path.join(destination_train_subdir, image)
            else:
                destination_path = os.path.join(destination_test, image)

            shutil.copyfile(source_path, destination_path)

if __name__ == "__main__":
    # Exemplo de uso
    source_directory = "editted_images"
    destination_directory = "image_sets"

    create_traintest_sets(source_directory, destination_directory)
