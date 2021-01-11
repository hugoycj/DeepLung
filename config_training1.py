config = {'train_data_path':['/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset0/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset1/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset2/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset3/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset4/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset5/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset6/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset7/',
                             '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset9/'],
          'val_data_path':['/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset8/'], 
          'test_data_path':['/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset8/'],  
          
          'train_preprocess_result_path':'/home/zhaojie/zhaojie/Lung/DeepLung-Minerva/Data/LUNA16PROPOCESSPATH/', # contains numpy for the data and label, which is generated by prepare.py
          'val_preprocess_result_path':'/home/zhaojie/zhaojie/Lung/DeepLung-Minerva/Data/LUNA16PROPOCESSPATH/',
          'test_preprocess_result_path':'/home/zhaojie/zhaojie/Lung/DeepLung-Minerva/Data/LUNA16PROPOCESSPATH/',
          
          'train_annos_path':'/home/zhaojie/zhaojie/Lung/data/luna16/CSVFILES/annotations.csv',
          'val_annos_path':'/home/zhaojie/zhaojie/Lung/data/luna16/CSVFILES/annotations.csv',
          'test_annos_path':'/home/zhaojie/zhaojie/Lung/data/luna16/CSVFILES/annotations.csv',

          'black_list':[],
          
          'preprocessing_backend':'python',
         } 